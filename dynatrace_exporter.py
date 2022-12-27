import argparse
import concurrent.futures
import gzip
import logging
import os
import re
import urllib.parse
import time
from typing import Callable, Dict
from wsgiref.simple_server import make_server

import numpy as np
import requests
import yaml

# Ignored dimensions of specific types for entity requests
IGNORED_ENTITY_DIMENSIONS_TYPES = ["List", "Map"]

GLOBAL_PREFIX = "dynatrace"
ENTITY_PREFIX = "entity"
METRIC_PREFIX = "metric"

# Dynatrace transformations to apply to all metrics
METRICS_TRANSFORMATIONS = ':lastReal:names'

# patterns to replace and their replacement in dimension names
DIMENSIONS_PATTERN_REPLACEMENTS = {
    'dt.entity.': ''
}

def lower_first_character(str: str):
    if not str:
        return str

    if len(str) == 1:
        return str.lower()
    
    return str[0].lower() + str[1:]

def escape_label_value(v: str):
    return "null" if not v else str(v).replace('"', '\\"')

def remove_special_characters(str: str):
    return re.sub("[\"'#%]", "", str)

def fmt_label_name(item: str):
    item = lower_first_character(item)
    item = remove_special_characters(item)
    return re.sub("[.:\- ]", '_', item) 

def fmt_metric_name(item: str):
    item = remove_special_characters(item)
    return re.sub("[.:\- ]", '_', item)

def first_non_null(values: list):
    if not values:
        return None

    for v in values:
        if v: 
            return v

    return None

def compile_regex(pattern: str):
    try:
        return re.compile(pattern)
    except:
        fatal_error(f"incorrect expression : {pattern}", 1)

def array_split(items: list, batch_size): 
    nb_batchs = int(len(items) / batch_size) + 1
    return np.array_split(list(items.keys()), nb_batchs)


##########################################################################
#### Config class helper
##########################################################################

# All valid entity types
CFG_VALID_ENTITY_TYPES = ["service", "application", "host"]

# Default exporter config values, using yaml paths
CFG_DEFAULT_VALUES = {
    "general.pageSize": 4000,
    "general.threads": 12,
    "collectors.service.enabled": True,
    "collectors.service.service_type": [],
    "collectors.application.enabled": True,
    "collectors.host.enabled": True,
    "collectors.metrics.enabled": True,
    "collectors.metrics.batchSize": 10,
    "collectors.metrics.deprecated": False,
    "collectors.metrics.whitelist": [],
    "collectors.metrics.params": {
        "resolution": 5
    }, 
    "logging.level": "INFO",
    "logging.format": '%(asctime)s %(levelname)s [%(threadName)s] - %(funcName)s() (%(filename)s#%(lineno)d) - %(message)s'
}

class Config(object):
    def __init__(self, cfg):
        self._cfg = cfg

    def cfg_value(self, yaml_path, required = False):
        keys = yaml_path.split(".")
        value = self._cfg
        for k in keys:
            if k not in value:
                if not required:
                    return CFG_DEFAULT_VALUES.get(yaml_path, None)
                else: 
                    fatal_error(f"Config error: could not find required property with path {yaml_path}. Key '{k}' does not exist")
                    
            value = value[k]

        return value

class LoggingConfig(Config):
    format: str
    level: str

    """
    Configuration class used to configure logging
    """
    def __init__(self, args: argparse.ArgumentParser, cfg: dict):
        """
        Parameters
        ----------
        `args`: argparse.ArgumentParse
            Command line arguments

        `cfg`: dict
            Configuration values, parsed from a YAML file for example
        """
        super().__init__(cfg)
        self.format = self.cfg_value("logging.format")
        self.level = "DEBUG" if args.debug else self.cfg_value("logging.level")

class DynatraceBadRequestException(Exception):
    url: str
    code: int
    message: str

    def __init__(self, url, response_error):
        super().__init__()
        self.url = url
        self.code = response_error["code"]
        self.message = response_error["message"]

class ExporterConfig(Config):
    threads: int
    page_size: int
    url_base: str
    api_key: str
    metric_enabled: bool
    metric_allow_deprecated: bool
    metrics_whitelist: list
    batch_size: int
    service_filters: list
    requested_entity_types: list
    metric_params: dict
    url_base: str

    def __init__(self, cfg):
        super().__init__(cfg)
        self.threads = max(self.cfg_value("general.threads"), 1)
        self.page_size = self.cfg_value("general.pageSize")
        self.url_base = self.cfg_value("general.api.url", required=True)
        self.api_key = self.cfg_value("general.api.apiKey", required=True)
        self.metric_enabled = self.cfg_value("collectors.metrics.enabled")
        self.metric_allow_deprecated = self.cfg_value("collectors.metrics.deprecated")
        self.metrics_whitelist = self.cfg_value("collectors.metrics.whitelist")
        self.batch_size = self.cfg_value("collectors.metrics.batchSize")
        self.service_filters = self.cfg_value("collectors.service.service_type")
        self.requested_entity_types = self.filter_entity_types()
        self.metric_params = self.build_metrics_url_params()

        if not self.metrics_whitelist and self.metric_enabled:
            logging.warning("Metrics are enabled but no whitelist configured. Fetching all metrics can take up to several minutes. Be careful to timeouts")

        # If url ends with a '/', remove it
        if self.url_base[-1] == '/':
            self.url_base = self.url_base[0:-1]

    # Build URL params to add to the metrics requests
    def build_metrics_url_params(self):
        mandatory_params = "fields=dimensionDefinitions,+unit,+description,displayName"
        additional_param = self.build_metrics_additional_url_params()
        return f"&{mandatory_params}&{additional_param}"

    # Build additionnal custom URL params for metrics requests. Based on config object 'collectors.metrics.params'
    def build_metrics_additional_url_params(self):
        params = self.cfg_value("collectors.metrics.params")
        return "&".join([f"{k}={v}" for k,v in params.items()])

    # Filter requested entity types to fetch
    def filter_entity_types(self):
        return list(filter(lambda t: self.cfg_value(f"collectors.{t}.enabled"), CFG_VALID_ENTITY_TYPES))


##########################################################################
#### Data Classes
##########################################################################

class SeriesItem(object):
    properties: dict
    value: float

    def __init__(self, properties: dict, value: float):
        self.properties = self.rename_dimensions(self.filter_dimensions(properties))
        self.value = value

    def __repr__ (self): return f"props={self.properties}, value={self.value}"
    
    def filter_dimensions(self, properties: dict) -> dict:
        result = dict()

        suffix = '.name'
        suffix_name_length = len(suffix)

        for k, v in properties.items():
            if not k.endswith('.name'):
                result[k] = v
            else:
                without_suffix = k[0:-suffix_name_length]
                if without_suffix not in properties or properties[without_suffix] != v:
                    result[k] = v

        return result

    def rename_dimensions(self, properties: dict) -> dict:
        result = dict()

        for k, v in properties.items():
            for pattern, replacement in DIMENSIONS_PATTERN_REPLACEMENTS.items():
                k = re.sub(pattern, replacement, k)

            result[k] = v

        return result

class Template(object):
    prometheus_type: str
    prometheus_help: str
    prometheus_name: str

    def __init__(self, prometheus_type, prometheus_help, prometheus_name):
        self.prometheus_type = prometheus_type
        self.prometheus_help = prometheus_help
        self.prometheus_name = prometheus_name

    def __repr__ (self): return f'name={self.prometheus_name}, type={self.prometheus_type}, help="{self.prometheus_help}"'

class EntityTemplate(Template):
    name: str
    series: str

    def __init__(self, dyn_entity_type: str):
        self.name = dyn_entity_type,
        self.series = [],

        prometheus_name = f"{GLOBAL_PREFIX}_{ENTITY_PREFIX}_{fmt_metric_name(dyn_entity_type)}_health"
        super().__init__("gauge", f"Health of entities of type '{dyn_entity_type}'", prometheus_name)

    def __repr__ (self): return f"{self.name} - series: {self.series}) - prometheus={{super()}}"

class MetricTemplate(Template):
    metricId: str
    unit: str

    def __init__(self, dyn_metric_name, unit, description): 
        self.metricId = dyn_metric_name,
        self.unit = unit,

        description = re.sub("[\n\r\t]+", "", description)

        prom_name = f"{GLOBAL_PREFIX}_{METRIC_PREFIX}_{fmt_metric_name(dyn_metric_name)}"
        super().__init__("gauge", f"({dyn_metric_name}) {description}", prom_name)

    def from_dynatrace_response(dic: dict):
        return MetricTemplate(dic["metricId"], dic["unit"], dic["description"])

    def __repr__ (self): return f"{self.metricId} ({self.unit})) - prometheus={{super()}}"


##########################################################################
#### Interfaces
##########################################################################


class CollectorInterface(object):
    def collect_entities(self):
        raise NotImplementedError()

    def collect_metrics(self):
        raise NotImplementedError()

class ExporterInterface(object):
    def export(self) -> str:
        raise NotImplementedError()
    

class DynatraceCollector(CollectorInterface):
    entity_gauges: dict
    metric_gauge: dict
    entity_properties: dict
    config: ExporterConfig

    def __init__(self, exporter_config: ExporterConfig):
        self.entity_gauges = dict()
        self.metric_gauge = dict()
        self.entity_properties = dict()
        self.config = exporter_config
        
        self.init_gauge_metric()
        self.init_gauge_entity()

    def init_gauge_entity(self):
        logging.info("Starting health metrics initialisation...")

        for entity_type in self.config.requested_entity_types:
            self.load_entity_properties(entity_type) 
            self.entity_gauges[entity_type] = EntityTemplate(entity_type)

        logging.debug("Health metrics initialized")

    def init_gauge_metric(self):
        if not self.config.metric_enabled:
            logging.info("Metrics have been disabled")
            return

        logging.info("Starting metric initialisation...")
        metric_ids = self.get_all_dynatrace_metrics()
        
        # No whitelist: all the metrics we found
        if not self.config.metrics_whitelist:
            self.metric_gauge = metric_ids
            # for metric_id in metric_ids:
            #     self.metric_gauge[metric_id] = metric_ids[metric_id]
            return

        # Whitelist: filter metrics using metric name or regex
        for whitelist_entry in self.config.metrics_whitelist:
            if whitelist_entry in metric_ids:
                # Metric exactly matching whitelist entry
                self.metric_gauge[whitelist_entry] = metric_ids[whitelist_entry]
            else:
                # Metric matching regex
                self.fill_metrics_matching_regex(metric_ids, whitelist_entry)
                

        logging.debug("Metrics initialized")

    def fill_metrics_matching_regex(self, metric_ids, whitelist_entry):
        regex = compile_regex(whitelist_entry)
        matching_metrics = list(filter(regex.match, metric_ids))

        if matching_metrics:
            logging.info(f"Retrieving metrics matching regex {whitelist_entry}")
            for m in matching_metrics:
                logging.info(f"Found matching metric: {m}")
                self.metric_gauge[m] = metric_ids[m]
        else:
            logging.warning(f"No metric matching the following whitelist regex entry: {whitelist_entry}")
            

    def call_api(self, url):
        try:
            response = requests.get(url, headers={"Authorization": self.config.api_key})
            self.check_dynatrace_response(response)

            return response.json()
        except requests.HTTPError:
            err = response.json()["error"]
            if err and err["code"] == 400:
                raise DynatraceBadRequestException(url, err)

            logging.error(f'HTTPError occurred during dynatrace HTTP request. code={err["code"]}, url={url}, msg="{err["message"]}"')
            return None
        except Exception as err:
            logging.error(f'Error occurred during dynatrace HTTP request. msg="{err}" URL={url}')
            return None

    def retry_call_api(self, url, max_attempts = 5):
        attempts = 0
        response = self.call_api(url)

        while not response and attempts < max_attempts:
            attempts += 1
            logging.warning("call_api() failed. Retry {}/{}. url={}".format(attempts, max_attempts, url))
            response = self.call_api(url)

        if response:
            return response

        raise Exception(f"Failed load dynatrace metrics. Max attempts ({max_attempts}) reached.")

    def check_dynatrace_response(self, response):
        response.raise_for_status()

        json = response.json()
        if "error" in json:
            msg = json["error"]["message"]
            code = json["error"]["code"]
            raise Exception(f'code={code} msg="{msg}"')


    def get_all_dynatrace_metrics(self):
        url = f"{self.config.url_base}/metrics?fields=dimensionDefinitions,+unit,+displayName,+description,pageSize={self.config.page_size}"

        metric_ids = dict()
        while True:
            response = self.retry_call_api(url, 5)
            self.fill_metrics(response, metric_ids)

            # If not page left, exit the loop    
            if self.was_last_page(response):
                break
            
            # Some pages left. Each API response provides a key to next page (nextPageKey)
            url = "{}/metrics?nextPageKey={}".format(self.config.url_base, response["nextPageKey"])    

        return metric_ids

    def fill_metrics(self, response_data: dict, dest: dict):
        if response_data and "metrics" in response_data:
            for metric in response_data["metrics"]:
                if self.ignore_metric(metric):
                    continue

                dest[metric["metricId"]] = MetricTemplate.from_dynatrace_response(metric)

    def ignore_metric(self, metric: dict) -> bool:
        if not self.config.metric_allow_deprecated:
            if self.is_metric_deprecated(metric):
                logging.debug(f'Ignoring metric {metric["metricId"]}: Metric is marked as deprecated in Dynatrace')
                return True

        return False

    def is_metric_deprecated(self, metric: dict) -> bool:
        if metric["displayName"].startswith("[Deprecated]"):
            return True
        if metric["description"].startswith("[Deprecated]"):
            return True
        if metric["displayName"].endswith("(deprecated)"):
            return True
        if metric["description"].endswith("(deprecated)"):
            return True

        return False


    def was_last_page(self, response_dict) -> bool:
        return response_dict and "nextPageKey" not in response_dict or response_dict["nextPageKey"] is None


    def load_entity_properties(self, entity_type: str):
        url = f"{self.config.url_base}/entityTypes/{entity_type}"
        response = self.retry_call_api(url, 5)

        if not response:
            return

        allowed_props = list(filter(lambda prop: prop["type"] not in IGNORED_ENTITY_DIMENSIONS_TYPES, response["properties"]))
        self.entity_properties[entity_type] = list(map(lambda prop: prop["id"], allowed_props))



    def collect_entities(self) -> dict:
        logging.info("Collecting health metrics...")
        entities = {
            entity_type: self.fetch_healthy_entities(entity_type) + self.fetch_unhealthy_entities(entity_type)
            for entity_type in self.entity_properties
        }

        counts = ', '.join(map(lambda k: f'{k}={len(entities[k])}' , entities.keys()))
        logging.debug("Health metrics collected. {}".format(counts))
        
        return entities

    def collect_metrics(self) -> dict:
        if not self.config.metric_enabled:
            return {}

        logging.info("Collecting standard metrics...")

        all_results = self.fetch_metrics_raw()
        metrics = {
            metric_id: self.build_metric_series_list(data)
            for metric_id, data in all_results.items()
        }

        logging.debug("Standard metrics collected. Total collected metrics: {}".format(len(metrics)))
        return metrics
    
    def build_metric_series_list(self, full_list):
        mapped = map(lambda item: SeriesItem(item["dimensionMap"], first_non_null(item["values"])), full_list)
        filtered = filter(lambda item: item.value is not None, mapped)
        return list(filtered)


    def fetch_metrics_raw(self) -> list:
        metric_id_batchs = array_split(self.metric_gauge, self.config.batch_size)

        if self.config.threads > 1:
            logging.debug("Multithreaded metrics fetching. nb_metrics={}, batch_size={}, batch_count={}, threads={}"
                .format(len(self.metric_gauge), self.config.batch_size, len(metric_id_batchs), self.config.threads))

            executor = concurrent.futures.ThreadPoolExecutor(self.config.threads)
            raw_results = executor.map(self.perform_batched_metrics_api_calls, metric_id_batchs)
        else:
            logging.debug("Single threaded metrics fetching. nb_metrics={}".format(len(self.metric_gauge)))
            raw_results = map(self.perform_batched_metrics_api_calls, metric_id_batchs)

        data = {}
        for batch_result in raw_results:
            for metric in batch_result:
                metric_name = metric['metricId'].replace(METRICS_TRANSFORMATIONS, '')
                data[metric_name] = metric["data"]

        return data

    def perform_batched_metrics_api_calls(self, batch_metric_ids):
        url = self.build_metrics_url(batch_metric_ids)

        try:
            response = self.call_api(url)

            if not response:
                logging.error(f'An error occurred during series fetching. URL="{url}", metric_ids={list(batch_metric_ids)}')
                return list()

            return response["result"]
        except DynatraceBadRequestException as ex:
            if ex.code == 400 and ex.message.startswith("Incompatible resolutions"):
                # If Incompatible resolution happens, we try to unbatch metrics and query one by one, we will get default resolution
                metriclist = ",".join(batch_metric_ids)
                logging.debug(f"Incompatible resolution error occurred. Metrics will be fetched one by one. Metrics=[{metriclist}]")
                return list(map(lambda m: self.perform_batched_metrics_api_calls([m])[0], batch_metric_ids))
            else:
                logging.error("Unknown Dynatrace Bad Request error: code={}, message={}, url={}".format(ex.code, ex.message, ex.url))
                return list()

    def build_metrics_url(self, metric_ids):
        queried_metrics = list(map(lambda x: x + METRICS_TRANSFORMATIONS, metric_ids))

        metric_ids_string = urllib.parse.quote(",".join(queried_metrics))
        return  "{}/metrics/query?metricSelector={}{}".format(self.config.url_base, metric_ids_string, self.config.metric_params)

    def build_entities_first_url(self, entity_type, health_type):
        properties = self.entity_properties.get(entity_type)
        props = ",".join(map(lambda p: f"+properties.{p}", properties))
        return '{}/entities?entitySelector=type("{}"),healthState("{}")&fields={}&pageSize={}'.format(
            self.config.url_base, entity_type, health_type, props, str(self.config.page_size))

    def build_entities_next_url(self, next_page_key):
        return "{}/entities?nextPageKey={}".format(self.config.url_base, next_page_key)

    def fetch_healthy_entities(self, entity_type):
        return self.fetch_entity_health_status(entity_type, 1)

    def fetch_unhealthy_entities(self, entity_type):
        return self.fetch_entity_health_status(entity_type, 0)

    def fetch_entity_health_status(self, entity_type, health_value):
        health_key = "HEALTHY" if health_value == 1 else "UNHEALTHY"
        url = self.build_entities_first_url(entity_type, health_key)

        series = []
        while True:
            response = self.call_api(url)
            self.fill_entities_health(response, series, entity_type, health_value)

            # If not page left, exit the loop    
            if self.was_last_page(response):
                break

            url = self.build_entities_next_url(response["nextPageKey"])
                    
        return series

    def fill_entities_health(self, response_data: dict, dest: list, entity_type: str, health_value: int):
        if not response_data or "entities" not in response_data:
            return

        for entity in response_data["entities"]:
            if self.is_entity_allowed(entity_type, entity):
                properties_dict = entity["properties"].copy()
                dest.append(SeriesItem(properties_dict, health_value))

    def is_entity_allowed(self, entity_type, entity) -> bool: 
        if not entity_type.lower() == "service":
            return True
        else:
            if not self.config.service_filters: # No filter => Allow all services
                return True
            elif entity["properties"]["serviceType"] in self.config.service_filters: # Filter => Only requested types
                return True
            else:
                return False
                

##########################################################################
#### Collector class
#### Gathers metrics and other data from Dynatrace
##########################################################################

class DynatraceExporter(ExporterInterface):
    collector: DynatraceCollector
    config: ExporterConfig

    def __init__(self, collector: DynatraceCollector, config: ExporterConfig):
        self.collector = collector
        self.config = config

    def export(self):
        start = time.time()

        metrics = self.collector.collect_metrics()
        fmt_metrics = self.format(self.collector.metric_gauge, metrics)

        entities = self.collector.collect_entities()
        fmt_entities = self.format(self.collector.entity_gauges, entities)

        result = fmt_metrics
        if result and result[-1] != '\n':
            result += "\n"
        result += fmt_entities

        logging.info("Collection phase finished. Total duration: {:.3f}s".format(time.time() - start))
        return result

    def format(self, templates: Dict[str, Template], values_map: dict): 
        lines = []
        for dyn_name, template in templates.items():
            prom_name = template.prometheus_name

            # Help and type
            lines.append(self.fmt_help(template))
            lines.append(self.fmt_type(template))

            # Series
            if dyn_name in values_map:
                for m in values_map[dyn_name]:
                    lines.append(self.fmt_series_item(prom_name, m))

        if not lines:
            return ""

        return "\n".join(lines) + "\n"

    def fmt_help(self, template: Template) -> str:
        return "# HELP {} {}".format(template.prometheus_name, template.prometheus_help)

    def fmt_type(self, template: Template) -> str:
        return "# TYPE {} {}".format(template.prometheus_name, template.prometheus_type)

    def fmt_series_item(self, prom_name: str, item: SeriesItem) -> str: 
        value = item.value
        labels = self.fmt_labels(item.properties)
        return f"{prom_name}{labels} {str(value)}"

    def fmt_labels(self, props) -> str: 
        if not props:
            return ""

        key_values = []
        for name, value in props.items():
            fmt_label = fmt_label_name(name)
            fmt_value = escape_label_value(value)

            key_values.append(f'{fmt_label}="{fmt_value}"')

        labels = ",".join(key_values)
        return '{' + labels + '}'

def load_cfg(path: str) -> dict:
    with open(path, "r") as stream:
        try:
            parsed = yaml.safe_load(stream)
            logging.info("Loaded configuration")
            return parsed
        except yaml.YAMLError as exc:
            fatal_error(f"Error occurred during YAML configuration parsing: {exc}", exc_info=True)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynatrace data exporter")

    parser.add_argument(
        "--config.file",
        dest="config_file",
        required=False,
        help="Path to config file",
        default=os.environ.get("CONFIG_FILE", "dynatrace_conf.yml"),
    )
    
    parser.add_argument(
        "--web.listen-address",
        dest="listen_address",
        required=False,
        help="Listen to this address and port",
        default=os.environ.get("LISTEN_ADDRESS", ":8000"),
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        required=False,
        action="store_true",
        help="Enable debug",
        default=os.environ.get("debug", 0) == 1,
    )
    
    parser.add_argument(
        "--test",
        dest="test",
        required=False,
        action="store_true",
        help="Test metrics collection once, then stop exporter",
        default=os.environ.get("TEST", 0) == 1,
    )
    
    parser.add_argument(
        "--output",
        dest="output",
        required=False,
        help="Output file when --test option is set",
        default=os.environ.get("CONFIG_FILE"),
    )
    return parser.parse_args()


def extract_port_and_address(listen_address):
    address, port = listen_address.split(":")
    address = address if address != "" else "0.0.0.0"
    return address, port


def bake_output(dynatrace_exporter: ExporterInterface, disable_compression: bool):
    output = dynatrace_exporter.export().encode("utf-8")
    headers = [('Content-Type', "text/plain")]
    if not disable_compression:
        output = gzip.compress(output)
        headers.append(('Content-Encoding', 'gzip'))
    
    return '200 OK', headers, output


def make_wsgi_app(dynatrace_exporter: ExporterInterface, disable_compression: bool = False) -> Callable:
    def exporter_app(environ, start_response):
        if environ['PATH_INFO'] == '/favicon.ico':
            status = '200 OK'
            headers = [('', '')]
            output = b''
        elif environ['PATH_INFO'] == '/metrics':
            status, headers, output = bake_output(dynatrace_exporter, disable_compression)
        else:
            status = '404 Not Found'
            headers = [('', '')]
            output = b''
        start_response(status, headers)
        return [output]
    return exporter_app

def configure_logging(cfg: LoggingConfig):
    logging.getLogger().setLevel(cfg.level)
    logging.getLogger().handlers[0].setFormatter(logging.Formatter(cfg.format))

def fatal_error(msg, code = 1, exc_info = False): 
    logging.critical(msg, exc_info = exc_info); 
    exit(code)


##########################################################################
#### Helper functions
##########################################################################

def run_test(args, exporter: DynatraceExporter):
    result = exporter.export()
    
    if args.output:
        with open(args.output, "w") as file:
            file.write(result)
        logging.info(f"Results written to file {args.output}")
    else:
        print("\n================== RESULT ==================\n")
        print(result)
        print("\n================== DONE ==================")


#### MAIN

args = parse_args()
cfg = load_cfg(args.config_file)

configure_logging(LoggingConfig(args, cfg))

exporter_cfg = ExporterConfig(cfg)
exporter = DynatraceExporter(DynatraceCollector(exporter_cfg), cfg)

if args.test:
    run_test(args, exporter)
    exit(0)

app = make_wsgi_app(exporter)

address, port = extract_port_and_address(args.listen_address)
httpd = make_server(address, int(port), app)

logging.info(f"Server sarting on {address}:{port}")
httpd.serve_forever()
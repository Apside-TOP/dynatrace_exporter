# Dynatrace Metric Exporter

Prometheus exporter for Dynatrace's metrics and entity monitoring.

Scrapes `/metrics` and `/entity` endpoints of **Dynatrace API v2** and exposes metrics to Prometheus.

This exporter allows you to easily retrieve all Dynatrace metrics and infrastructure heatlth state (healthy/unhealthy applications, hosts and services).

Exporter developped by [Apside TOP](https://www.apside.com/) and sponsored by [Harmonie Mutuelle](https://www.harmonie-mutuelle.fr/).


<div align="center">
  <img alt="Apside TOP" src="assets/logo_apside_top.png" height="100" align="center" />
  &emsp;&emsp;
  <img alt="Harmonie Mutuelle" src="assets/logo_hm.svg" height="140" align="center" />
</div>

# Grafana visualisations example 

Apdex score, active users and errors per application :

![Grafana Table](assets/grafana_table_apdex.png?raw=true "Apdex score, active users and errors per application")

Applications health state :

![Applications health state](assets/grafana_honeycomb_application.png?raw=true "Applications health state")

***Note**: This panel uses [Hexmap](https://grafana.com/grafana/plugins/marcusolsson-hexmap-panel/) plugin.*

# Command line

```
❯ python3 dynatrace_exporter.py -h
usage: dynatrace_exporter.py [-h] [--config.file CONFIG_FILE] [--web.listen-address LISTEN_ADDRESS] [--debug] [--test] [--output OUTPUT]

Dynatrace data exporter

optional arguments:
  -h, --help            show this help message and exit
  --config.file CONFIG_FILE
                        Path to config file (default: dynatrace_exporter.yml)
  --web.listen-address LISTEN_ADDRESS
                        Listen to this address and port (default: :9126)
  --debug               Enable debug (default: False)
  --test                Test metrics collection once, then stop exporter (default: False)
  --output OUTPUT       Output file when --test option is set (default: None)
```

# Running with a virtualenv

1. Create and activate a virtual env

Inside project folder: 

```sh
# Initialize you venv
python3 -m venv .venv

# Activate your venv
source .venv/bin/activate

# To confirm the virtual environment is activated
# python executable whould be inside .env directory
which python

# Check venv python version
python --version

# Install dependencies inside your venv:
pip install -r requirements.txt
```

To deactivate your virtual env:
```sh
deactivate
```

2. Create a file `dynatrace_exporter.yml` with your configuration

3. Run the application (venv must be active)
```sh
python ./dynatrace_exporter.py --config.file dynatrace_exporter.yml
```

You can now navigate to http://localhost:9126/metrics.

# Running with docker

1. Build the image
```she
docker build . -t <your_image>
```

2. Create a file `dynatrace_exporter.yml` with your configuration
3. Run the image

```sh
docker run \
    -v ./dynatrace_exporter.yml:/usr/src/app/dynatrace_exporter.yml \
    -p 9126:9126
    <your_image>
```

You can now navigate to http://localhost:9126/metrics.

# Configuration

```yml
# General configurations
general:
  api:
    # API base URL. Should end with "/api/v2"
    url: "https://xxxxxxxxxx/api/v2"
    # Dynatrace API key
    apiKey: "Api-Token xxxxxxxxxx"
    # Additional custom headers for HTTP requests
    headers:
      # <header name>: <header value>
  
  # Number of parallel threads to use to fetch metrics. 1 or lower to disable multithreading
  threads: 12 
  # Max number of results to return in one request. Max allowed by dynatrace is 4000
  pageSize: 4000

# Collectors configuration
collectors:
  # Collecting health of Dynatrace entities with type 'service'
  service:
    enabled: true
    # Service's type whitelist. Empty or undefined to allow all types of service
    # service_type:
    #   - "DATABASE_SERVICE"
      
  # Collecting health of Dynatrace entities with type 'application'
  application:
    enabled: true

  # Collecting health of Dynatrace entities with type 'host'
  host:
    enabled: true

  # Collecting Dynatrace metrics
  metrics:
    enabled: true
    # Maximal number of metrics fetched to batch fetch in each requet. Max allowed by dynatrace is 10
    batchSize: 10 
    # Allow deprecated metrics
    deprecated: false
    # Additionnal URL parameters. See full list here: https://www.dynatrace.com/support/help/dynatrace-api/environment-api/metric-v2/get-data-points 
    params:
      # Number of points to retrieve
      resolution: 12 
      # Time range of query
      from: now-6h 
    # Metrics regex whitelist. Empty or undefined to allow all metrics. 
    # Warning: enabling all (or at least a lot of) metrics might slow the exporter down to several seconds, or even minutes. 
    # Be careful to timeouts on Prometheus side.
    whitelist:
      - builtin:apps.+ 
      # - builtin:tech.+ 
      # - builtin:service.+ 
      # - builtin:host.+ 
      # - builtin:cloud.+ 
      # - builtin:pgi.+ 
      # - builtin:containers.+ 
      # - builtin:kubernetes.+ 
      # - builtin:billing.+ 
      # - builtin:synthetic.+ 
      # - builtin:queue.+ 
      # - builtin:security.+ 
      # - builtin:span.+ 
      # - builtin:dashboards.+ 
      # - builtin:process.+ 
      # - builtin:osservice.+ 
      # - calc:.+ 
      # - dsfm:.+ 
      # - ext:.+ 
      # - func:.+ 

# Logging configurations
logging:
  # Minimal log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  level: INFO 
  # Log format. To see all available attributes: https://docs.python.org/3/library/logging.html#logrecord-attributes 
  format: "%(asctime)s %(levelname)s [%(threadName)s] - (%(filename)s#%(lineno)d) - %(message)s"
```

# Metrics

```bash
# HELP dynatrace_metric_<metric_name> <description>
# TYPE dynatrace_metric_<metric_name> <dynatrace_metric_name> gauge
dynatrace_metric_<metric_name>{<labels>}<value>
# HELP dynatrace_entity_service_health service_health
# TYPE dynatrace_entity_service_health gauge 
dynatrace_entity_service_health{<labels>}<value>
# HELP dynatrace_entity_host_health host_health
# TYPE dynatrace_entity_host_health gauge 
dynatrace_entity_host_health{<labels>}<value>
# HELP dynatrace_entity_application_health application_health
# TYPE dynatrace_entity_application_health gauge 
dynatrace_entity_application_health{<labels>}<value>
```

# Sponsors

Harmonie Mutuelle (https://www.harmonie-mutuelle.fr/) 

![Logo Harmonie Mutuelle](assets/logo_hm.svg "Logo Harmonie Mutuelle")


# Maintainers

* [[Apside TOP](https://www.apside.com/)] Robin Maréchal ([@robinmarechal](https://github.com/robinmarechal))

# License

Apache License 2.0, see [LICENSE](LICENSE).

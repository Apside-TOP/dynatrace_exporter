#!/bin/bash

ROOT=$(dirname "$0")

INCLUDED_FILES="dynatrace_exporter.py dynatrace_exporter.yml requirements.txt LICENSE README.md"

RELEASE_BRANCH_PREFIX="release/"
RELEASE_BRANCH_PREFIX_REGEX="${RELEASE_BRANCH_PREFIX/\//\\\/}"
SEMVER_REGEX="(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-(alpha|beta)[0-9]*)?"

CMD_CREATE_VERSION=0
CMD_LIST_VERSIONS=0
CMD_LATEST_VERSION=0

VERSION=""
TYPE=""
LIST_LIMIT=20

show_help() {
    cat << EOM
Small script to help handling versions and release.

It helps you creating versions by automatically incrementing major, minor or patch part of the version, or creating a version of your choice.
Version must follows semantic versionning pattern (X.Y.Z, X=major, Y=minor, Z=patch).
It will create the branch 'release/X.Y.Z', the tag 'vX.Y.Z' and both zip and tar.gz archives.
   
Usage: 
$ROOT/release.sh [-h|--help] [-l|--last] [-l|--list] [-c|--create <MAJOR|MINOR|PATCH|version>]

  -h --help               Display this help
  --last                  Print most recent version
  -l --list               List already created versions. use --limit to change max number of displayed versions
      --limit $LIST_LIMIT              Limit number of version to print with option --list
  -c --create <version>   Create a new version. 
                          Valid values are : MAJOR, MINOR, PATCH or a version number following semantic versioning format x.y.z
                          - MAJOR: skip to next major version. Eg: if last version is 1.2.3, version 2.0.0 will be created
                          - MINOR: skip to next minor version. Eg: if last version is 1.2.3, version 1.3.0 will be created
                          - PATCH: increment patch version by one. Eg: if last version is 1.2.3, version 1.2.4 will be created
                          After creating the branch, the tag and the archives, you should create the yourself on GitHub, selecting the newly created tag.
EOM
}

echoeval(){
  echo ">>> $*"
  eval "$*"
}

get_part(){
  echo $1 | cut -d . -f $2
}

get_major(){
  get_part $1 1
}

get_minor(){
  get_part $1 2
}

get_patch(){
  get_part $1 3
}

safe_inc_patch(){
  patch_version="$1"

  if [[ "$patch_version" =~ [0-9]+\-(alpha|beta)[0-9]* ]]; then
    patch_value=$(echo $patch_version | cut -d - -f 1)
    alphabeta=$(echo $patch_version | grep -oP "(alpha|beta)")
    numeric_value=$(echo $patch_version | grep -oP [0-9]$)

    if [[ "$numeric_value" == "" ]]; then 
      numeric_value=2
    else
      numeric_value=$(( $numeric_value + 1 ))
    fi

    echo "$patch_value-$alphabeta$numeric_value"
  else
    echo $(( $patch_version + 1 ))
  fi
}

fetch_last_versions(){
  limit="$1"
  # git branch -a | grep -oP "${RELEASE_BRANCH_PREFIX}${SEMVER_REGEX}$" | sort -Vr | uniq | head -$limit | sed s+"${RELEASE_BRANCH_PREFIX}"+''+ | grep -oP $SEMVER_REGEX
  git branch -a | grep -oP "${RELEASE_BRANCH_PREFIX}${SEMVER_REGEX}$" | sort -Vr | uniq | head -$limit | grep -oP $SEMVER_REGEX
}

inc_version_major(){
  last=$(fetch_last_versions 1)

  major=$(get_major $last)
  inc=$(($major + 1))

  echo "$inc.0.0"
}

inc_version_minor(){
  last=$(fetch_last_versions 1)

  major=$(get_major $last)
  minor=$(get_minor $last)
  inc=$(($minor + 1))

  echo "$major.$inc.0"
}

inc_version_patch(){
  last=$(fetch_last_versions 1)

  major=$(get_major $last)
  minor=$(get_minor $last)
  patch=$(get_patch $last)
  inc=$(safe_inc_patch $patch)

  echo "$major.$minor.$inc"
}

handle_version_value(){
  original="$VERSION"

  if [[ ! "$VERSION" =~ (MAJOR|MINOR|PATCH|$SEMVER_REGEX) ]]; then
    echo "ERROR: Version format is invalid: $VERSION (original value was: $original)"
    exit 1
  fi

  if [[ "$VERSION" =~ (MAJOR|MINOR|PATCH) ]] && [[ "$(fetch_last_versions 1)" == "" ]]; then
    echo "ERROR: No published version yet. Cannot increment $VERSION. Please provide a version to create the first one."
    exit 1
  fi

  if [[ "$VERSION" == "MAJOR" ]]; then 
    VERSION=$(inc_version_major)
  elif [[ "$VERSION" == "MINOR" ]]; then  
    VERSION=$(inc_version_minor)
  elif [[ "$VERSION" == "PATCH" ]]; then  
    VERSION=$(inc_version_patch)
  else
    existing="$(git branch -a | grep -oP "${RELEASE_BRANCH_PREFIX_REGEX}${VERSION}$")"

    if [[ "$existing" != "" ]]; then
      echo "ERROR: version $VERSION already exists."
      echo "Please use --list to see already existing versions."
      exit 1
    fi
  fi
}

get_archive_name(){
  echo "dynatrace_exporter-$VERSION"
}

_cmd_list_versions(){
  echo "Listing last $LIST_LIMIT versions:"
  fetch_last_versions $LIST_LIMIT
}

_cmd_show_last_version(){
  echo "Latest version: "
  fetch_last_versions 1
}

_cmd_create_version(){
  current_branch=$(git rev-parse --abbrev-ref HEAD)

  handle_version_value

  _cmd_show_last_version
  read -p "Version $VERSION will be created. Continue ? [Y/n]: " response

  if [[ "$response" != ""  ]] && [[ ! "$response" =~ [yY] ]]; then
    echo "User response was: $response. Aborting operation."
    exit 0
  fi

  remote=$(git remote | head -1)

  echo "Version $VERSION will be created."
  echo ""
  echo "Updating local repository..."
  echoeval git pull
  echo ""

  branch_name=$RELEASE_BRANCH_PREFIX$VERSION
  echo "Creating branch $branch_name..."
  echoeval git checkout -b $branch_name
  echo ""

  tag="v$VERSION"
  echo "Creating tag $tag..."
  echoeval git tag $tag
  echo ""

  echo "Publishing branch and tag to remote repository..."
  echoeval git push $remote $branch_name
  echoeval git push --tags
  echo ""

  archive_name=$(get_archive_name)
  zipfile="$archive_name.zip"
  targzfile="$archive_name.tar.gz"
  echo "Creating archives $zipfile and $targzfile..."
  echoeval zip $archive_name.zip $INCLUDED_FILES
  echoeval tar -czf $archive_name.tar.gz $INCLUDED_FILES
  echo ""

  echo "returning back to previous branch $current_branch..."
  git checkout $current_branch
  
  echo ""
  echo "DONE!"
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      show_help
      exit 0
      ;;
    last)
      CMD_LATEST_VERSION=1
      ;;
    list)
      CMD_LIST_VERSIONS=1
      ;;
    --limit)
      LIST_LIMIT="$2"
      shift
      ;;
    create)
      CMD_CREATE_VERSION=1
      VERSION="${2^^}" # ^^ => String to upper
      shift
      ;;
    -*|--*)
      echo "Unknown option $1"
      exit 1
      ;;
  esac
  shift
done

if [[ $CMD_LATEST_VERSION == 1 ]]; then
  _cmd_show_last_version
elif [[ $CMD_LIST_VERSIONS == 1 ]]; then
  _cmd_list_versions
elif [[ $CMD_CREATE_VERSION == 1 ]]; then
  _cmd_create_version
else
  show_help
fi
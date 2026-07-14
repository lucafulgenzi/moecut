#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/hf-gguf-download.sh [options] [model-url-or-repo-or-search]

Downloads one GGUF file from a Hugging Face model repository.

Requirements:
  bash 4+, jq, mkdir, and curl or wget.

Options:
  -o, --out DIR       Output directory. Defaults to the current directory.
  -r, --revision REV  Hugging Face revision/branch. Defaults to main.
  -f, --file FILE     Download this exact GGUF file from the repository.
  -t, --token TOKEN   Hugging Face token. Defaults to HF_TOKEN when set.
  -y, --yes           Download the first GGUF match without prompting.
  -h, --help          Show this help.

Examples:
  scripts/hf-gguf-download.sh unsloth/Qwen3-30B-A3B-GGUF
  scripts/hf-gguf-download.sh https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF
  scripts/hf-gguf-download.sh "Qwen3 30B GGUF"
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

need_fetcher() {
    if command -v curl >/dev/null 2>&1; then
        FETCHER="curl"
    elif command -v wget >/dev/null 2>&1; then
        FETCHER="wget"
    else
        die "need curl or wget to talk to Hugging Face"
    fi
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

check_requirements() {
    if [ -z "${BASH_VERSINFO+x}" ] || [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
        die "need Bash 4 or newer"
    fi

    require_command jq
    require_command mkdir
    need_fetcher
}

urlencode() {
    local input="${1}"
    local i ch out=""
    local old_lc="${LC_ALL:-}"
    export LC_ALL=C

    for ((i = 0; i < ${#input}; i++)); do
        ch="${input:i:1}"
        case "$ch" in
            [a-zA-Z0-9.~_-]) out+="$ch" ;;
            /) out+="/" ;;
            *) printf -v out '%s%%%02X' "$out" "'$ch" ;;
        esac
    done

    if [ -n "$old_lc" ]; then
        export LC_ALL="$old_lc"
    else
        unset LC_ALL
    fi
    printf '%s' "$out"
}

urldecode() {
    local input="${1//+/ }"
    printf '%b' "${input//%/\\x}"
}

show_file_info_value() {
    local value="$1"
    local query part

    case "$value" in
        *\?*) query="${value#*\?}" ;;
        *) return 1 ;;
    esac

    while [ -n "$query" ]; do
        part="${query%%&*}"
        if [ "$part" = "$query" ]; then
            query=""
        else
            query="${query#*&}"
        fi

        case "$part" in
            show_file_info=*)
                urldecode "${part#show_file_info=}"
                return 0
                ;;
        esac
    done

    return 1
}

api_get() {
    local url="$1"

    if [ "$FETCHER" = "curl" ]; then
        if [ -n "$TOKEN" ]; then
            curl -fsSL -H "Authorization: Bearer $TOKEN" "$url"
        else
            curl -fsSL "$url"
        fi
    else
        if [ -n "$TOKEN" ]; then
            wget -qO- --header="Authorization: Bearer $TOKEN" "$url"
        else
            wget -qO- "$url"
        fi
    fi
}

download_file() {
    local url="$1"
    local out="$2"

    if [ "$FETCHER" = "curl" ]; then
        if [ -n "$TOKEN" ]; then
            curl -L --fail --continue-at - -H "Authorization: Bearer $TOKEN" \
                --output "$out" "$url"
        else
            curl -L --fail --continue-at - --output "$out" "$url"
        fi
    else
        if [ -n "$TOKEN" ]; then
            wget -c --header="Authorization: Bearer $TOKEN" -O "$out" "$url"
        else
            wget -c -O "$out" "$url"
        fi
    fi
}

normalize_repo() {
    local value="$1"

    value="${value#https://huggingface.co/}"
    value="${value#http://huggingface.co/}"
    value="${value%%\?*}"
    value="${value%%#*}"
    value="${value%%/tree/*}"
    value="${value%%/resolve/*}"
    value="${value%%/blob/*}"
    value="${value%/}"

    printf '%s' "$value"
}

looks_like_repo() {
    local value
    value="$(normalize_repo "$1")"
    [[ "$value" =~ ^[^[:space:]/]+/[^[:space:]/]+$ ]]
}

quant_label() {
    local file="$1"
    local upper="${file^^}"

    if [[ "$upper" =~ (^|[^A-Z0-9])(IQ[0-9]_[A-Z0-9_]+|Q[0-9]_[A-Z0-9_]+|BF16|F16|F32)([^A-Z0-9_]|$) ]]; then
        printf '%s' "${BASH_REMATCH[2]}"
    else
        printf 'unknown'
    fi
}

choose_number() {
    local prompt="$1"
    local max="$2"
    local choice

    while true; do
        printf '%s' "$prompt" >&2
        read -r choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max" ]; then
            printf '%s' "$choice"
            return 0
        fi
        printf 'Choose a number from 1 to %s.\n' "$max" >&2
    done
}

search_repo() {
    local query="$1"
    local encoded count i choice
    local results=()

    encoded="$(urlencode "$query")"
    mapfile -t results < <(api_get "https://huggingface.co/api/models?search=$encoded&limit=20" | jq -r '.[].modelId // empty')

    [ "${#results[@]}" -gt 0 ] || die "no Hugging Face models found for '$query'"

    printf '\nModels:\n' >&2
    for i in "${!results[@]}"; do
        printf '%2d) %s\n' "$((i + 1))" "${results[$i]}" >&2
    done

    count="${#results[@]}"
    if [ "$YES" -eq 1 ]; then
        printf '%s\n' "${results[0]}"
    else
        choice="$(choose_number 'Choose model number: ' "$count")"
        printf '%s\n' "${results[$((choice - 1))]}"
    fi
}

show_ggufs() {
    local i

    printf '\nAvailable GGUF files:\n'
    for i in "${!FILES[@]}"; do
        printf '%2d) %-10s %s\n' "$((i + 1))" "$(quant_label "${FILES[$i]}")" "${FILES[$i]}"
    done
}

fetch_ggufs() {
    local repo="$1"
    local revision="$2"
    local url

    url="https://huggingface.co/api/models/$(urlencode "$repo")?revision=$(urlencode "$revision")"
    mapfile -t FILES < <(api_get "$url" | jq -r '.siblings[]?.rfilename // empty | select(test("\\.gguf$"; "i"))')
}

file_exists_in_list() {
    local needle="$1"
    local file

    for file in "${FILES[@]}"; do
        if [ "$file" = "$needle" ]; then
            return 0
        fi
    done

    return 1
}

OUT_DIR="."
REVISION="main"
TOKEN="${HF_TOKEN:-}"
YES=0
FETCHER=""
INPUT=""
REQUESTED_FILE=""
FILES=()

while [ "$#" -gt 0 ]; do
    case "$1" in
        -o|--out)
            [ "$#" -ge 2 ] || die "$1 needs a directory"
            OUT_DIR="$2"
            shift 2
            ;;
        -r|--revision)
            [ "$#" -ge 2 ] || die "$1 needs a revision"
            REVISION="$2"
            shift 2
            ;;
        -f|--file)
            [ "$#" -ge 2 ] || die "$1 needs a GGUF filename"
            REQUESTED_FILE="$2"
            shift 2
            ;;
        -t|--token)
            [ "$#" -ge 2 ] || die "$1 needs a token"
            TOKEN="$2"
            shift 2
            ;;
        -y|--yes)
            YES=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [ -n "$INPUT" ]; then
                die "only one model URL, repo id, or search string is accepted"
            fi
            INPUT="$1"
            shift
            ;;
    esac
done

[ "$#" -eq 0 ] || die "unexpected extra arguments: $*"

check_requirements

if [ -z "$INPUT" ]; then
    printf 'Hugging Face model URL, repo id, or search: '
    read -r INPUT
fi

[ -n "$INPUT" ] || die "empty input"

if [ -z "$REQUESTED_FILE" ]; then
    REQUESTED_FILE="$(show_file_info_value "$INPUT" || true)"
fi

if looks_like_repo "$INPUT"; then
    REPO="$(normalize_repo "$INPUT")"
else
    REPO="$(search_repo "$INPUT")"
fi

printf '\nRepository: %s\n' "$REPO"

fetch_ggufs "$REPO" "$REVISION"
[ "${#FILES[@]}" -gt 0 ] || die "no GGUF files found in $REPO at revision $REVISION"

show_ggufs

COUNT="${#FILES[@]}"
if [ -n "$REQUESTED_FILE" ]; then
    if ! file_exists_in_list "$REQUESTED_FILE"; then
        die "requested GGUF file not found in $REPO: $REQUESTED_FILE"
    fi
    FILE="$REQUESTED_FILE"
elif [ "$YES" -eq 1 ]; then
    FILE="${FILES[0]}"
else
    CHOICE="$(choose_number 'Choose GGUF number: ' "$COUNT")"
    FILE="${FILES[$((CHOICE - 1))]}"
fi

[ -n "$FILE" ] || die "invalid GGUF choice"

mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${FILE##*/}"
URL="https://huggingface.co/$REPO/resolve/$(urlencode "$REVISION")/$(urlencode "$FILE")"

printf '\nDownloading %s\n' "$FILE"
printf 'Output: %s\n' "$OUT"
download_file "$URL" "$OUT"
printf '\nSaved %s\n' "$OUT"

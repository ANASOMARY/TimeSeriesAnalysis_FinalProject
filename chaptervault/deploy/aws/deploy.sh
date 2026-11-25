#!/bin/bash
# Installs or updates ChapterVault on an Amazon Linux 2023 server.
#
# Run as root. Safe to run again at any time: it fetches the latest code, rebuilds the image and
# restarts the two containers. Uploaded chapters, accounts and certificates live in Docker volumes
# and are not touched.
#
# Reads /opt/chaptervault/config:
#   REPO       git URL of the repository
#   REF        branch or tag to deploy
#   DOMAIN     address reviewers use (blank = automatic <ip>.sslip.io address)
#   PUBLIC_IP  the server's fixed public IP address
set -euo pipefail

BASE=/opt/chaptervault
export HOME="${HOME:-/root}"   # unset when run through Systems Manager; git wants one

# Everything lives in main() so the whole script is parsed before the code checkout below
# replaces this very file.
main() {
    # shellcheck disable=SC1091
    . "$BASE/config"
    : "${REPO:?missing in config}" "${REF:?missing in config}" "${PUBLIC_IP:?missing in config}"
    DOMAIN="${DOMAIN:-}"
    local host="${DOMAIN:-$(echo "$PUBLIC_IP" | tr . -).sslip.io}"

    echo "== Installing Docker"
    if ! command -v docker >/dev/null || ! command -v git >/dev/null; then
        dnf install -y docker git
    fi
    systemctl enable --now docker

    echo "== Fetching the code ($REF)"
    if [ -d "$BASE/src/.git" ]; then
        git -C "$BASE/src" fetch --depth 1 origin "$REF"
        git -C "$BASE/src" checkout -f FETCH_HEAD
    else
        git clone --depth 1 --branch "$REF" "$REPO" "$BASE/src"
    fi

    echo "== Building the image (several minutes the first time)"
    local libreoffice=yes
    if ! docker build -t chaptervault "$BASE/src/chaptervault"; then
        echo "!! Full build failed; building without LibreOffice"
        libreoffice=no
        docker build --build-arg WITH_LIBREOFFICE=0 -t chaptervault "$BASE/src/chaptervault"
    fi

    # One fixed setup token, so the link stays valid across restarts until the author account exists.
    if [ ! -s "$BASE/setup-token" ]; then
        (umask 077 && openssl rand -hex 24 > "$BASE/setup-token")
    fi
    local token
    token="$(cat "$BASE/setup-token")"

    echo "== Starting ChapterVault at https://$host"
    docker network inspect chaptervault >/dev/null 2>&1 || docker network create chaptervault >/dev/null
    docker rm -f chaptervault-app chaptervault-web >/dev/null 2>&1 || true
    docker run -d --name chaptervault-app --restart unless-stopped \
        --network chaptervault --network-alias app \
        -v chaptervault-data:/data \
        -e CHAPTERVAULT_BEHIND_PROXY=1 \
        -e CHAPTERVAULT_BASE_URL="https://$host" \
        -e CHAPTERVAULT_SETUP_TOKEN="$token" \
        chaptervault >/dev/null
    # Caddy terminates HTTPS and obtains and renews the certificate on its own.
    docker run -d --name chaptervault-web --restart unless-stopped \
        --network chaptervault \
        -p 80:80 -p 443:443 \
        -v chaptervault-certs:/data \
        "${CADDY_IMAGE:-caddy:2}" caddy reverse-proxy --from "${CADDY_FROM:-$host}" --to app:8000 >/dev/null

    echo "== Waiting for the app to answer"
    local ready=no
    for _ in $(seq 1 40); do
        if docker exec chaptervault-app python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/login', timeout=3)" 2>/dev/null; then
            ready=yes
            break
        fi
        sleep 3
    done
    if [ "$ready" != yes ]; then
        echo "!! The app did not start:"
        docker logs --tail 30 chaptervault-app || true
        exit 1
    fi

    local note="ChapterVault is running."
    [ "$libreoffice" = yes ] || note="$note Word files are re-flowed: LibreOffice could not be installed."
    (umask 077 && printf 'SETUP_URL=%q\nNOTE=%q\n' "https://$host/setup?token=$token" "$note" > "$BASE/result")
    echo "== Done. $note"
    echo "   Address: https://$host"
}

main "$@"

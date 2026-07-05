#!/usr/bin/env bash
set -euo pipefail

INSTALL_DIR="${HOME}/bin"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${INSTALL_DIR}"

ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64|amd64)
    RCLONE_ARCH="amd64"
    ;;
  aarch64|arm64)
    RCLONE_ARCH="arm64"
    ;;
  armv7l)
    RCLONE_ARCH="arm-v7"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}"
    echo "Download manually from https://rclone.org/downloads/"
    exit 1
    ;;
esac

ZIP_NAME="rclone-current-linux-${RCLONE_ARCH}.zip"
URL="https://downloads.rclone.org/${ZIP_NAME}"

echo "Installing rclone without sudo"
echo "Architecture: ${ARCH} -> ${RCLONE_ARCH}"
echo "Download URL: ${URL}"
echo "Install directory: ${INSTALL_DIR}"

cd "${TMP_DIR}"
curl -L -o "${ZIP_NAME}" "${URL}"
python -m zipfile -e "${ZIP_NAME}" .

RCLONE_BIN="$(find . -type f -name rclone | head -n 1)"
if [[ -z "${RCLONE_BIN}" ]]; then
  echo "Could not find rclone binary after extracting ${ZIP_NAME}"
  exit 1
fi

cp "${RCLONE_BIN}" "${INSTALL_DIR}/rclone"
chmod +x "${INSTALL_DIR}/rclone"

case ":${PATH}:" in
  *":${INSTALL_DIR}:"*)
    echo "${INSTALL_DIR} is already in PATH"
    ;;
  *)
    echo ""
    echo "Add this line to your shell profile if it is not already there:"
    echo "export PATH=\"\$HOME/bin:\$PATH\""
    echo ""
    if [[ -f "${HOME}/.bashrc" ]]; then
      if ! grep -q 'export PATH="$HOME/bin:$PATH"' "${HOME}/.bashrc"; then
        echo 'export PATH="$HOME/bin:$PATH"' >> "${HOME}/.bashrc"
        echo "Added PATH update to ${HOME}/.bashrc"
      fi
    fi
    export PATH="${INSTALL_DIR}:${PATH}"
    ;;
esac

echo ""
echo "rclone installed successfully:"
"${INSTALL_DIR}/rclone" version

echo ""
echo "For this terminal session, run:"
echo "export PATH=\"\$HOME/bin:\$PATH\""

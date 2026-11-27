#!/bin/bash
# Run in AWS CloudShell to move the ChapterVault site to an address of your own.
#
#   curl -fsSL https://raw.githubusercontent.com/ANASOMARY/TimeSeriesAnalysis_FinalProject/ccr-ad86fbe7-utaqxg/chaptervault/deploy/aws/set-address.sh | bash -s -- chaptervault.example.org
#
# Before running it, create a DNS "A" record for that address pointing at the server's IP address.
# Chapters, accounts, notes and settings are kept. The previous address stops working.
set -euo pipefail

DOMAIN="${1:-}"
STACK="${2:-chaptervault}"
if ! [[ "$DOMAIN" =~ ^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$ ]]; then
    echo "Give the new address in lower case, for example: chaptervault.example.org"
    exit 1
fi

read -r SERVER ADDRESS <<<"$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$STACK" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].[InstanceId,PublicIpAddress]" --output text)"
if [ -z "${SERVER:-}" ] || [ "$SERVER" = "None" ]; then
    echo "No running server named '$STACK' was found in this region. Check the region shown at the top right."
    exit 1
fi

# The certificate can only be issued once the name really points at the server.
POINTS_TO=$(getent ahostsv4 "$DOMAIN" | awk 'NR==1 {print $1}' || true)
if [ "$POINTS_TO" != "$ADDRESS" ] && [ "${SKIP_DNS_CHECK:-}" != "1" ]; then
    echo "$DOMAIN does not point at the server yet."
    echo "  It resolves to: ${POINTS_TO:-nothing}"
    echo "  It should be:   $ADDRESS"
    echo "Create a DNS A record for $DOMAIN with the value $ADDRESS (switch off any proxy option for it),"
    echo "wait a few minutes, and run this command again."
    exit 1
fi

echo "Moving server $SERVER to https://$DOMAIN. This takes a few minutes..."
# shellcheck disable=SC2016  # $status is for the server to expand, not this shell
COMMANDS=$(printf '{"commands":[
  "export HOME=/root",
  "sed -i \\"s/^DOMAIN=.*/DOMAIN=%s/\\" /opt/chaptervault/config",
  "bash /opt/chaptervault/src/chaptervault/deploy/aws/deploy.sh > /var/log/chaptervault-update.log 2>&1; status=$?",
  "grep -E \\"^(==|!!)\\" /var/log/chaptervault-update.log",
  "[ $status -eq 0 ] || tail -n 15 /var/log/chaptervault-update.log",
  "exit $status"
],"executionTimeout":["1800"]}' "'$DOMAIN'")
COMMAND=$(aws ssm send-command --instance-ids "$SERVER" --document-name AWS-RunShellScript \
    --comment "Move ChapterVault to $DOMAIN" --parameters "$COMMANDS" --query Command.CommandId --output text)

while :; do
    sleep 6
    STATE=$(aws ssm get-command-invocation --command-id "$COMMAND" --instance-id "$SERVER" \
        --query Status --output text 2>/dev/null || echo Pending)
    case "$STATE" in
        Pending|InProgress|Delayed) printf '.' ;;
        *) break ;;
    esac
done
echo
aws ssm get-command-invocation --command-id "$COMMAND" --instance-id "$SERVER" \
    --query StandardOutputContent --output text
if [ "$STATE" != "Success" ]; then
    echo "The change did not finish (status: $STATE)."
    aws ssm get-command-invocation --command-id "$COMMAND" --instance-id "$SERVER" \
        --query StandardErrorContent --output text
    exit 1
fi

echo "Waiting for the secure certificate for $DOMAIN..."
for _ in $(seq 1 30); do
    if curl -fsS -o /dev/null --max-time 8 "https://$DOMAIN/login" 2>/dev/null; then
        echo "Done. Your site is now at https://$DOMAIN"
        echo "Sign in again there. Invitation links sent with the old address no longer work: resend them."
        exit 0
    fi
    sleep 6
done
echo "The server has been moved, but https://$DOMAIN is not answering yet."
echo "The certificate can take a few more minutes. Try the address in your browser shortly."

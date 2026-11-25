#!/bin/bash
# Run in AWS CloudShell to update the ChapterVault server to the latest code.
# Chapters, accounts, notes and settings are kept.
#
#   curl -fsSL https://raw.githubusercontent.com/ANASOMARY/TimeSeriesAnalysis_FinalProject/ccr-ad86fbe7-utaqxg/chaptervault/deploy/aws/update.sh | bash
set -euo pipefail

STACK="${1:-chaptervault}"
SERVER=$(aws ec2 describe-instances \
    --filters "Name=tag:Name,Values=$STACK" "Name=instance-state-name,Values=running" \
    --query "Reservations[0].Instances[0].InstanceId" --output text)
if [ -z "$SERVER" ] || [ "$SERVER" = "None" ]; then
    echo "No running server named '$STACK' was found in this region. Check the region shown at the top right."
    exit 1
fi

echo "Updating server $SERVER. This takes a few minutes..."
# The server runs its own install script again and reports the milestones (or, on failure, the last log lines).
# shellcheck disable=SC2016  # $status is for the server to expand, not this shell
COMMANDS='{"commands":[
  "export HOME=/root",
  "bash /opt/chaptervault/src/chaptervault/deploy/aws/deploy.sh > /var/log/chaptervault-update.log 2>&1; status=$?",
  "grep -E \"^(==|!!)\" /var/log/chaptervault-update.log",
  "[ $status -eq 0 ] || tail -n 15 /var/log/chaptervault-update.log",
  "exit $status"
],"executionTimeout":["1800"]}'
COMMAND=$(aws ssm send-command --instance-ids "$SERVER" --document-name AWS-RunShellScript \
    --comment "Update ChapterVault" --parameters "$COMMANDS" --query Command.CommandId --output text)

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
if [ "$STATE" = "Success" ]; then
    echo "Updated. Reload the site in your browser."
else
    echo "The update did not finish (status: $STATE). The site keeps running the previous version if the build failed."
    aws ssm get-command-invocation --command-id "$COMMAND" --instance-id "$SERVER" \
        --query StandardErrorContent --output text
    exit 1
fi

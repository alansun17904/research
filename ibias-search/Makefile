REMOTE := kennel:/data/cl/u/awsun/rttt

.PHONY: sync logs

# Push local source while excluding environments, caches, and outputs.
sync:
	rsync -az --itemize-changes --exclude-from=.rsyncignore ./ $(REMOTE)/

# Download experiment logs into the local project.
logs:
	rsync -az $(REMOTE)/logs/ ./cluster-logs/

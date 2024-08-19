#!/bin/bash

# Check if the SSH key is provided
if [ -z "$SSH_KEY" ]; then
  echo "Error: SSH_KEY environment variable is not set."
  exit 1
fi


if [ ! -f "/root/.ssh/id_rsa" ]; then
  # Copy the SSH key from the environment variable
  echo "${SSH_KEY}" > ~/.ssh/id_rsa
  chmod 600 ~/.ssh/id_rsa

  # Start SSH agent and add the key
  eval $(ssh-agent -s)
  ssh-add ~/.ssh/id_rsa

  # Add GitHub to known hosts to avoid interactive prompt
  ssh-keyscan -H github.com >> ~/.ssh/known_hosts
fi


if [ ! -d "/work/auto-bidding/.git" ]; then
  # Clone the repository
  mkdir -p /work/temp
  git clone git@github.com:DanielGardin/auto-bidding.git /work/temp
  shopt -s dotglob
  rsync -a /work/temp/* /work/auto-bidding
  rm -rf /work/temp

  export PYTHONPATH="/root/biddingTrainEnv:${PYTHONPATH}"
fi

# Run the main process
exec "$@"

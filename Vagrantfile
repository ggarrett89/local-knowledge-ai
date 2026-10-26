# -*- mode: ruby -*-
# local-knowledge-ai: Ubuntu VM to run the API, ChromaDB and Ollama.
# Independent of any other repository. Does not download models or install Ollama.
#
# Resources adjustable without editing this file:
#   LKA_VM_CPUS=4 LKA_VM_MEMORY_MB=8192 vagrant up

cpus   = Integer(ENV.fetch("LKA_VM_CPUS", "4"))
memory = Integer(ENV.fetch("LKA_VM_MEMORY_MB", "8192"))

Vagrant.configure("2") do |config|
  # Official Ubuntu 22.04 box (Python 3.10). If it is not in the local cache yet,
  # Vagrant downloads it on the first `vagrant up`.
  config.vm.box = "ubuntu/jammy64"
  config.vm.hostname = "local-knowledge-ai"

  # API reachable only from the host itself (127.0.0.1).
  config.vm.network "forwarded_port", guest: 8000, host: 8000, host_ip: "127.0.0.1", id: "api"

  # Project code. Index data does NOT live here (see LKA_DATA_DIR in docs/vm-plan.md).
  config.vm.synced_folder ".", "/vagrant", type: "virtualbox"

  config.vm.provider "virtualbox" do |vb|
    vb.name   = "local-knowledge-ai"
    vb.cpus   = cpus
    vb.memory = memory
    vb.gui    = false
  end

  # Basic packages only. Ollama and models are installed manually (see docs/vm-plan.md).
  config.vm.provision "shell", inline: <<-SHELL
    set -eu
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -y
    apt-get install -y --no-install-recommends python3-venv python3-pip git curl ca-certificates sqlite3
    install -d -o vagrant -g vagrant /home/vagrant/lka-data
  SHELL
end

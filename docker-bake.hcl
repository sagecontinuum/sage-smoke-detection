target "plugin-base" {
  dockerfile = "Dockerfile"
  tags       = ["plugin-base:latest"]
}

target "notebook" {
  dockerfile = "Dockerfile.notebook"
  contexts = {
    plugin-base = "target:plugin-base"
  }
  tags = ["notebook:latest"]
}
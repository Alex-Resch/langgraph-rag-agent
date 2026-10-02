variable "region" {
  type    = string
  default = "eu-central-1"
}

variable "aws_profile" {
  description = "AWS CLI profile with write access, see `aws configure --profile`."
  type        = string
  default     = "terraform-deploy"
}

variable "project_name" {
  type    = string
  default = "langgraph-rag-agent"
}

variable "budget_alert_email" {
  description = "Gets an email as soon as the monthly AWS costs exceed the budget."
  type        = string
}

variable "monthly_budget_usd" {
  type    = string
  default = "1"
}

variable "container_port" {
  type    = number
  default = 7860
}

variable "cpu" {
  description = "Fargate CPU units (1024 = 1 vCPU)."
  type        = number
  default     = 1024
}

variable "memory" {
  description = "Fargate memory in MiB."
  type        = number
  default     = 4096
}

variable "app_secrets" {
  description = "API keys passed to the container as environment variables, e.g. from .env."
  type        = map(string)
  sensitive   = true
}

variable "github_repository" {
  description = "Only this repository's main branch may deploy, as owner/name."
  type        = string
  default     = "Alex-Resch/langgraph-rag-agent"
}

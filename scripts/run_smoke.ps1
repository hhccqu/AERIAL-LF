param([string]$Config = "configs/local.toml")
$ErrorActionPreference = "Stop"
python -m aerial_lf.audit --config $Config --require-checkpoints
python -m aerial_lf.train --config $Config --smoke


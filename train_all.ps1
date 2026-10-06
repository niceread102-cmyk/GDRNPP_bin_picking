# Train the detector, then the pose network (one at a time, each capped by gpu_mem_fraction in the config).
# Logs: runs\detector.log, runs\pose.log. Extra args go to both scripts, e.g.  --set pose.train.epochs=60
Set-Location $PSScriptRoot
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
New-Item -ItemType Directory -Force runs | Out-Null
$extra = $args -join " "

# cmd handles the redirection: PowerShell 5.1 would turn Python's stderr warnings into errors
cmd /c ".venv\Scripts\python.exe -u train_detector.py $extra > runs\detector.log 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Error "detector training failed (see runs\detector.log)"; exit 1 }
cmd /c ".venv\Scripts\python.exe -u train_pose.py $extra > runs\pose.log 2>&1"
if ($LASTEXITCODE -ne 0) { Write-Error "pose training failed (see runs\pose.log)"; exit 1 }

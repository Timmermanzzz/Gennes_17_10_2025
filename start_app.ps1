# Ensure we run from the project root
$projectPath = "C:\Users\rens_\Documents\VisualStudio\Rienk\BU gennes 3\gennes5 Dubbel Druppel v2\gennesV3 Dubbel Druppel\Gennes_17_10_2025"
Set-Location -Path $projectPath

# Free common Streamlit ports if occupied
$ports = @(8501, 8599)
foreach ($p in $ports) {
  try {
    Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  } catch { }
}

# Create logs directory
$logDir = Join-Path $projectPath ".streamlit-logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $logDir ("streamlit_out_" + $timestamp + ".log")
$stderr = Join-Path $logDir ("streamlit_err_" + $timestamp + ".log")

# Launch Streamlit (uses .streamlit/config-local.toml for local development)
$arguments = '-m streamlit run app.py --config .streamlit/config-local.toml'
$proc = Start-Process -FilePath "python" -ArgumentList $arguments -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
Write-Host "Started Streamlit (PID $($proc.Id)). Logs: `n $stdout `n $stderr"

# Brief wait and report port status
Start-Sleep -Seconds 2
netstat -ano | findstr ":8501" | Out-Host

# Quick Fix for "unsupported gloo device" Error

## The Problem

The error `makeDeviceForHostname(): unsupported gloo device` occurs because Gloo is trying to resolve hostnames (like "MuditLaptop") instead of using IP addresses directly.

## The Solution

**CRITICAL**: You must ensure `GLOO_SOCKET_IFNAME` is **completely unset** (not just set to `$null`). Also, make sure you're using **IP addresses only**, never hostnames.

### On BOTH Devices (Master and Worker):

```powershell
# 1. Remove GLOO_SOCKET_IFNAME completely if it exists
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue

# 2. Set all required variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's actual IP (NOT hostname!)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"

# 3. Set rank-specific variables
# On Master:
$env:RANK = "0"
$env:LOCAL_RANK = "0"

# On Worker:
$env:RANK = "1"
$env:LOCAL_RANK = "0"

# 4. Verify GLOO_SOCKET_IFNAME is NOT set
if ($env:GLOO_SOCKET_IFNAME) {
    Write-Host "ERROR: GLOO_SOCKET_IFNAME is still set to: $env:GLOO_SOCKET_IFNAME"
    Write-Host "Remove it with: Remove-Item Env:\GLOO_SOCKET_IFNAME"
} else {
    Write-Host "OK: GLOO_SOCKET_IFNAME is not set"
}
```

## Important Notes

1. **Never use hostnames** - Always use IP addresses for `MASTER_ADDR`
2. **GLOO_SOCKET_IFNAME must be unset** - Don't set it to an IP address or hostname
3. **Both devices use the same MASTER_ADDR** - The master's IP address
4. **Verify with ping** - Both devices should be able to ping each other using IPs

## Finding Your IP Address

```powershell
# Get your IP address
ipconfig | Select-String "IPv4"

# Or more specifically:
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
    $_.IPAddress -like "172.*" -or 
    $_.IPAddress -like "192.*" -or 
    $_.IPAddress -like "10.*"
} | Select-Object IPAddress, InterfaceAlias
```

## Complete Setup Script

### Master Device (Rank 0)

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Clean up any existing GLOO_SOCKET_IFNAME
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue

# Set environment variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # YOUR master IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"

# Verify
Write-Host "MASTER_ADDR: $env:MASTER_ADDR"
Write-Host "GLOO_SOCKET_IFNAME: $(if ($env:GLOO_SOCKET_IFNAME) { $env:GLOO_SOCKET_IFNAME } else { 'NOT SET (GOOD)' })"

# Run training
python src/train_ddp.py --config experiments/config_jester.yaml
```

### Worker Device (Rank 1)

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Clean up any existing GLOO_SOCKET_IFNAME
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue

# Set environment variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"

# Verify
Write-Host "MASTER_ADDR: $env:MASTER_ADDR"
Write-Host "GLOO_SOCKET_IFNAME: $(if ($env:GLOO_SOCKET_IFNAME) { $env:GLOO_SOCKET_IFNAME } else { 'NOT SET (GOOD)' })"

# Run training (start within 5-10 seconds of master)
python src/train_ddp.py --config experiments/config_jester.yaml
```

## If It Still Doesn't Work

1. **Check Windows Firewall**: Allow port 29500 (and 29501) for both inbound and outbound
2. **Test connectivity**: `Test-NetConnection -ComputerName 172.20.10.2 -Port 29500`
3. **Check both devices are on same network**: `ipconfig` should show same subnet
4. **Try disabling firewall temporarily** to test if that's the issue


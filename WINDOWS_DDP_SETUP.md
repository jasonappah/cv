# Windows DDP Setup Guide

This guide helps you set up distributed training on Windows with the Gloo backend.

## Common Issues and Solutions

### 1. "unsupported gloo device" Error

**Cause**: `GLOO_SOCKET_IFNAME` is set to an IP address instead of a network interface name, or the interface configuration is incorrect.

**Solution**: 
- **Option A (Recommended)**: Unset `GLOO_SOCKET_IFNAME` and let Gloo auto-detect:
  ```powershell
  $env:GLOO_SOCKET_IFNAME = $null
  # or
  Remove-Item Env:\GLOO_SOCKET_IFNAME
  ```

- **Option B**: Set it to the actual network interface name (not IP address):
  ```powershell
  # First, find your interface name:
  Get-NetAdapter | Select-Object Name, InterfaceDescription, Status
  
  # Then set it (example):
  $env:GLOO_SOCKET_IFNAME = "Ethernet"
  # or
  $env:GLOO_SOCKET_IFNAME = "Wi-Fi"
  ```

### 2. Connection Errors (10049, 11004)

**Cause**: 
- `MASTER_ADDR` is set to `0.0.0.0` (invalid)
- Network connectivity issues
- Firewall blocking the port

**Solution**:
1. **Set MASTER_ADDR to the actual IP address of the master device**:
   ```powershell
   # On Master Device (Rank 0):
   $env:MASTER_ADDR = "172.20.10.2"  # Use the master's actual IP
   
   # On Worker Device (Rank 1):
   $env:MASTER_ADDR = "172.20.10.2"  # Use the master's IP (same as above)
   ```

2. **Verify network connectivity**:
   ```powershell
   # On worker device, ping the master:
   ping 172.20.10.2
   ```

3. **Check Windows Firewall**:
   - Allow inbound connections on port 29500
   - Or temporarily disable firewall for testing

### 3. IPv6 Address Retrieval Errors

**Cause**: Gloo is trying to use IPv6 when IPv4 is expected.

**Solution**: Ensure these are set:
```powershell
$env:GLOO_SOCKET_FAMILY = "INET"  # Force IPv4
$env:USE_LIBUV = "0"              # Disable libuv
```

## Complete Setup for Two Devices

### Master Device (Rank 0)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set environment variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # YOUR MASTER DEVICE'S IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
# DO NOT set GLOO_SOCKET_IFNAME, or set it to interface name (not IP)

# Run training
python src/train_ddp.py --config experiments/config_jester.yaml
```

### Worker Device (Rank 1)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Set environment variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # MASTER DEVICE'S IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
# DO NOT set GLOO_SOCKET_IFNAME, or set it to interface name (not IP)

# Run training
python src/train_ddp.py --config experiments/config_jester.yaml
```

## Finding Your IP Address

On Windows, find your IP address:

```powershell
# Get IPv4 address
ipconfig | Select-String "IPv4"

# Or more detailed:
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "172.*" -or $_.IPAddress -like "192.*" -or $_.IPAddress -like "10.*"}
```

## Quick Troubleshooting Checklist

- [ ] `MASTER_ADDR` is set to the actual IP of the master (not `0.0.0.0`)
- [ ] Both devices use the same `MASTER_ADDR` value
- [ ] `GLOO_SOCKET_IFNAME` is either unset or set to interface name (not IP)
- [ ] `GLOO_SOCKET_FAMILY` is set to `"INET"`
- [ ] `USE_LIBUV` is set to `"0"`
- [ ] Both devices can ping each other
- [ ] Windows Firewall allows port 29500
- [ ] Both devices are on the same network
- [ ] `WORLD_SIZE` matches the number of devices (2 in your case)
- [ ] `RANK` is unique for each device (0 for master, 1 for worker)

## Testing Connectivity

Before running training, test connectivity:

```powershell
# On worker device:
Test-NetConnection -ComputerName 172.20.10.2 -Port 29500
```

If this fails, check firewall settings.


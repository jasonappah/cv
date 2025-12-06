# Interface Name Auto-Detection Fix

## What Changed

The code now **automatically detects and sets the network interface name** to prevent Gloo from trying to resolve hostnames (which causes the "makeDeviceForHostname" error).

## How It Works

1. **Detects your local IP address** automatically
2. **Gets the network interface name** from that IP (e.g., "Ethernet", "Wi-Fi", "vEthernet")
3. **Sets GLOO_SOCKET_IFNAME** to that interface name
4. **Prevents hostname resolution** - Gloo uses the interface name instead of trying to resolve "MuditLaptop"

## What You Need to Do

### On BOTH Devices:

**You can now simply run:**

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Set environment variables (GLOO_SOCKET_IFNAME will be auto-detected)
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"  # or "1" on worker
$env:LOCAL_RANK = "0"

# Run training
python src/train_ddp.py --config experiments/config_jester.yaml
```

**You no longer need to manually set or remove GLOO_SOCKET_IFNAME** - the code handles it automatically!

## What You'll See

When you run the script, you should see:

```
Detected local IP: 172.20.10.2
Auto-detected interface name: Wi-Fi
Using GLOO_SOCKET_IFNAME: Wi-Fi
Rank 0: Initializing distributed training...
  Backend: gloo
  MASTER_ADDR: 172.20.10.2 (using IP directly)
  ...
```

## If Auto-Detection Fails

If you see:
```
WARNING: Could not auto-detect interface name. Gloo may try hostname resolution.
```

You can manually set the interface name:

```powershell
# First, find your interface name:
Get-NetAdapter | Select-Object Name, InterfaceDescription, Status

# Then set it (example):
$env:GLOO_SOCKET_IFNAME = "Wi-Fi"
# or
$env:GLOO_SOCKET_IFNAME = "Ethernet"
```

## Manual Interface Name Setup (Optional)

If you prefer to set it manually:

```powershell
# Find your interface name
Get-NetAdapter | Select-Object Name

# Set it (use the exact name from above)
$env:GLOO_SOCKET_IFNAME = "Wi-Fi"  # or "Ethernet", etc.
```

## Complete Setup Scripts

### Master Device (Rank 0)

```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # YOUR master IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

### Worker Device (Rank 1)

```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

## Key Benefits

✅ **No more manual GLOO_SOCKET_IFNAME management**  
✅ **Prevents hostname resolution errors**  
✅ **Works automatically on both devices**  
✅ **Falls back gracefully if detection fails**

Try running the training again - the interface name should be auto-detected and the "makeDeviceForHostname" error should be resolved!


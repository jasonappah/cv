# Final Fix for DDP Training Issues

## Changes Made to Code

1. **Removed TCPStore** - It was causing libuv errors. Now using `init_method` directly (more reliable for Gloo).
2. **Set USE_LIBUV=0 at the very top** - Before any torch imports to ensure it's respected.
3. **Auto-resolve hostnames** - If MASTER_ADDR is a hostname, it's automatically resolved to IP.
4. **Better GLOO_SOCKET_IFNAME handling** - Automatically removes it if set to an IP address.

## What You Need to Do

### On BOTH Devices (Master and Worker):

```powershell
# 1. Activate virtual environment
.\venv\Scripts\Activate.ps1

# 2. CRITICAL: Remove GLOO_SOCKET_IFNAME completely
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue

# 3. Set all environment variables
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's actual IP address
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"

# 4. Set rank-specific variables
# On Master Device:
$env:RANK = "0"
$env:LOCAL_RANK = "0"

# On Worker Device:
$env:RANK = "1"
$env:LOCAL_RANK = "0"

# 5. Verify GLOO_SOCKET_IFNAME is NOT set
if ($env:GLOO_SOCKET_IFNAME) {
    Write-Host "ERROR: GLOO_SOCKET_IFNAME is still set!"
    Remove-Item Env:\GLOO_SOCKET_IFNAME
} else {
    Write-Host "OK: GLOO_SOCKET_IFNAME is not set"
}

# 6. Run training
python src/train_ddp.py --config experiments/config_jester.yaml
```

## Key Points

1. **USE_LIBUV=0** - Must be set (the code now sets it automatically, but set it in your environment too)
2. **GLOO_SOCKET_IFNAME** - Must be completely unset (not just `$null`)
3. **MASTER_ADDR** - Use IP address, not hostname (the code will auto-resolve if you use hostname, but IP is better)
4. **Both devices use same MASTER_ADDR** - The master's IP address

## Execution Order

1. **Start Master Device (Rank 0) first**
2. **Wait 2-3 seconds**
3. **Start Worker Device (Rank 1)**

## If You Still Get "makeDeviceForHostname" Error

This error means Gloo is trying to resolve a hostname. Try these additional steps:

### Option 1: Check Windows Hostname Resolution

```powershell
# Check your hostname
hostname

# Check if hostname resolves correctly
nslookup $(hostname)
```

### Option 2: Temporarily Disable Hostname in hosts file

If your hostname "MuditLaptop" is causing issues, you can add an entry to force it to resolve to your IP:

```powershell
# Get your IP address
ipconfig | Select-String "IPv4"

# Edit hosts file (run as Administrator)
notepad C:\Windows\System32\drivers\etc\hosts

# Add this line (replace with your actual IP):
# 172.20.10.2    MuditLaptop
```

### Option 3: Use IP Address for Everything

Make sure you're using IP addresses everywhere, never hostnames:
- `MASTER_ADDR` = IP address
- No hostname references in any environment variables

## Testing Connectivity

Before running training, test that both devices can communicate:

```powershell
# On worker device, test connection to master
Test-NetConnection -ComputerName 172.20.10.2 -Port 29500

# Should show: TcpTestSucceeded : True
```

If this fails, check Windows Firewall settings.

## Complete Setup Scripts

### Master Device (Rank 0) - Copy and Paste

```powershell
.\venv\Scripts\Activate.ps1
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # YOUR master IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

### Worker Device (Rank 1) - Copy and Paste

```powershell
.\venv\Scripts\Activate.ps1
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

## What Changed in the Code

- ✅ Removed TCPStore (was causing libuv errors)
- ✅ Set USE_LIBUV=0 at module level (before torch imports)
- ✅ Auto-resolves hostnames to IPs if needed
- ✅ Automatically removes GLOO_SOCKET_IFNAME if set to IP
- ✅ Better error messages with troubleshooting tips

The code should now work better on Windows!


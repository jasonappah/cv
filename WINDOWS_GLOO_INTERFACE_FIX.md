# Fix for "makeDeviceForInterface(): unsupported gloo device"

## The Problem

Gloo on Windows has very limited support for network interfaces. Even when you set `GLOO_SOCKET_IFNAME` to a valid interface name like "Wi-Fi", Gloo may not be able to use it, causing the "unsupported gloo device" error.

## Solution: Let Gloo Auto-Detect

The code has been updated to **completely unset GLOO_SOCKET_IFNAME** and let Gloo auto-detect the interface. This is often more reliable on Windows.

## What You Need to Do

### On BOTH Devices:

1. **Make sure GLOO_SOCKET_IFNAME is NOT set** in your environment:
```powershell
# Remove it if it exists
Remove-Item Env:\GLOO_SOCKET_IFNAME -ErrorAction SilentlyContinue

# Verify it's not set
if ($env:GLOO_SOCKET_IFNAME) {
    Write-Host "ERROR: GLOO_SOCKET_IFNAME is still set!"
} else {
    Write-Host "OK: GLOO_SOCKET_IFNAME is not set"
}
```

2. **Set your environment variables:**
```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP (replace with actual)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"  # or "1" on worker
$env:LOCAL_RANK = "0"
```

3. **Update hosts file** (if you haven't already):
```powershell
# Run PowerShell as Administrator
notepad C:\Windows\System32\drivers\etc\hosts

# Add (replace with actual IPs):
172.20.10.2    MuditLaptop
172.20.10.3    rekhi-simar

# Flush DNS
ipconfig /flushdns
```

4. **Run training:**
```powershell
python src/train_ddp.py --config experiments/config_jester.yaml
```

## What Changed in the Code

The code now:
- **Automatically unsets GLOO_SOCKET_IFNAME** to let Gloo auto-detect
- This prevents the "unsupported gloo device" error
- Gloo will use the default interface automatically

## Alternative: If Auto-Detection Still Fails

If you still get errors, try manually setting the interface using its **index** instead of name:

```powershell
# Find interface index
Get-NetAdapter | Select-Object Name, InterfaceIndex

# Try setting by index (example - use actual index from above)
$env:GLOO_SOCKET_IFNAME = "1"  # Interface index, not name
```

But first, try with GLOO_SOCKET_IFNAME completely unset - that's usually the most reliable on Windows.

## Troubleshooting

If you still get "unsupported gloo device":

1. **Check Windows Firewall:**
   - Allow port 29500 for both inbound and outbound
   - Or temporarily disable firewall for testing

2. **Verify network connectivity:**
   ```powershell
   # On secondary device
   Test-NetConnection -ComputerName 172.20.10.2 -Port 29500
   ping 172.20.10.2
   ```

3. **Try different network interface:**
   - If using Wi-Fi, try Ethernet (or vice versa)
   - Some interfaces work better than others with Gloo

4. **Check if both devices are on same network:**
   ```powershell
   # Both should show same subnet
   ipconfig | Select-String "IPv4"
   ```

The updated code should now work better by letting Gloo auto-detect the interface!


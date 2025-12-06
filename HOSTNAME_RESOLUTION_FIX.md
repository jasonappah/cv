# Fix for Hostname Resolution Issue

## The Problem

Gloo is doing **reverse DNS lookups** - even though you set `MASTER_ADDR` to an IP address (172.20.10.3), Gloo internally resolves it to a hostname ("rekhi-simar" or "MuditLaptop") and tries to use that hostname, which causes connection failures.

## The Solution

You need to **add entries to the hosts file** on both devices to map IP addresses to hostnames. This prevents Gloo from doing reverse DNS lookups that fail.

### Step 1: Find IP Addresses on Both Devices

**On Master Device:**
```powershell
ipconfig | Select-String "IPv4"
# Note the IP address (e.g., 172.20.10.3)
hostname
# Note the hostname (e.g., MuditLaptop)
```

**On Secondary Device:**
```powershell
ipconfig | Select-String "IPv4"
# Note the IP address (e.g., 172.20.10.2)
hostname
# Note the hostname (e.g., rekhi-simar)
```

### Step 2: Edit Hosts File on BOTH Devices

**On Master Device (Run PowerShell as Administrator):**

```powershell
# Open hosts file
notepad C:\Windows\System32\drivers\etc\hosts

# Add these lines at the end (replace with your actual IPs and hostnames):
172.20.10.3    MuditLaptop
172.20.10.2    rekhi-simar
```

**On Secondary Device (Run PowerShell as Administrator):**

```powershell
# Open hosts file
notepad C:\Windows\System32\drivers\etc\hosts

# Add these lines at the end (replace with your actual IPs and hostnames):
172.20.10.3    MuditLaptop
172.20.10.2    rekhi-simar
```

**Important:** Both devices need the SAME entries in their hosts files!

### Step 3: Verify Hostname Resolution

**On both devices, test:**

```powershell
# Test forward lookup (hostname to IP)
nslookup MuditLaptop
nslookup rekhi-simar

# Test reverse lookup (IP to hostname)
nslookup 172.20.10.3
nslookup 172.20.10.2
```

### Step 4: Run Training Again

After updating the hosts file, try running training again:

**Master Device:**
```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.3"  # Master's IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

**Secondary Device:**
```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.3"  # Master's IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

## Alternative: Disable Reverse DNS (Advanced)

If editing hosts file doesn't work, you can try disabling reverse DNS lookups, but this is more complex and may not work with Gloo.

## Why This Happens

Gloo's C++ backend does reverse DNS lookups to identify network interfaces. On Windows, if the reverse lookup fails or returns an unexpected hostname, Gloo can't properly bind to the interface, causing the "requested address is not valid" error.

By adding entries to the hosts file, we ensure that:
1. Forward lookups (hostname → IP) work correctly
2. Reverse lookups (IP → hostname) return the expected hostname
3. Gloo can properly identify and use the network interfaces

## Troubleshooting

If you still get errors after updating hosts file:

1. **Verify hosts file was saved correctly:**
   ```powershell
   Get-Content C:\Windows\System32\drivers\etc\hosts
   ```

2. **Flush DNS cache:**
   ```powershell
   ipconfig /flushdns
   ```

3. **Test connectivity:**
   ```powershell
   # On secondary device
   Test-NetConnection -ComputerName 172.20.10.3 -Port 29500
   ping 172.20.10.3
   ```

4. **Check Windows Firewall:**
   - Ensure port 29500 is allowed for both inbound and outbound connections
   - Or temporarily disable firewall for testing


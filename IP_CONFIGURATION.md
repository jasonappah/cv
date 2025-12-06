# Correct IP Configuration

## Current Situation

- **Secondary Device (rekhi-simar)**: `172.20.10.3`
- **Master Device**: Need to find its IP

## The Problem

The secondary device is trying to connect to itself because `MASTER_ADDR` is set to its own IP (`172.20.10.3`). It should be set to the **master device's IP** instead.

## Solution

### Step 1: Find Master Device's IP

**On the Master Device (MuditLaptop), run:**
```powershell
ipconfig | Select-String "IPv4"
```

This will show the master's IP address (probably something like `172.20.10.2` or similar).

### Step 2: Update Hosts File on BOTH Devices

**On Master Device (run PowerShell as Administrator):**
```powershell
notepad C:\Windows\System32\drivers\etc\hosts
```

Add:
```
172.20.10.2    MuditLaptop
172.20.10.3    rekhi-simar
```
(Replace `172.20.10.2` with the actual master IP you found)

**On Secondary Device (run PowerShell as Administrator):**
```powershell
notepad C:\Windows\System32\drivers\etc\hosts
```

Add the same entries:
```
172.20.10.2    MuditLaptop
172.20.10.3    rekhi-simar
```
(Replace `172.20.10.2` with the actual master IP)

### Step 3: Flush DNS on Both Devices

```powershell
ipconfig /flushdns
```

### Step 4: Set Correct Environment Variables

**On Master Device (MuditLaptop):**
```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # MASTER'S OWN IP (replace with actual)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

**On Secondary Device (rekhi-simar):**
```powershell
.\venv\Scripts\Activate.ps1
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # MASTER'S IP (same as master uses, NOT 172.20.10.3!)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
python src/train_ddp.py --config experiments/config_jester.yaml
```

## Key Point

- **Master device**: `MASTER_ADDR` = its own IP (e.g., `172.20.10.2`)
- **Secondary device**: `MASTER_ADDR` = master's IP (e.g., `172.20.10.2`), NOT its own IP (`172.20.10.3`)

Both devices should use the **same MASTER_ADDR value** (the master's IP).


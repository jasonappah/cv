# How to Start DDP Training

## Step-by-Step Instructions

### Prerequisites
- ✅ Environment variables are set on both devices (as per WINDOWS_DDP_SETUP.md)
- ✅ Both devices are on the same network
- ✅ Both devices can ping each other
- ✅ Virtual environment is activated on both devices

### Execution Order

**IMPORTANT**: Start the **MASTER device (Rank 0) FIRST**, then start the **WORKER device (Rank 1)** within a few seconds.

---

## Step 1: Start Master Device (Rank 0)

On the **master device**, run:

```powershell
python src/train_ddp.py --config experiments/config_jester.yaml
```

**What to expect:**
- You should see diagnostic output showing:
  - `Rank 0: Initializing distributed training...`
  - Backend, MASTER_ADDR, MASTER_PORT, etc.
  - `Rank 0: Successfully initialized process group!`
- Then it will wait for the worker device to connect
- Once connected, training will begin automatically

**If you see errors:**
- Check that all environment variables are set correctly
- Verify `MASTER_ADDR` is your actual IP (not 0.0.0.0)
- Make sure `GLOO_SOCKET_IFNAME` is unset or set to interface name

---

## Step 2: Start Worker Device (Rank 1)

**Within 5-10 seconds** of starting the master, on the **worker device**, run:

```powershell
python src/train_ddp.py --config experiments/config_jester.yaml
```

**What to expect:**
- You should see diagnostic output:
  - `Rank 1: Initializing distributed training...`
  - Connection to master device
  - `Rank 1: Successfully initialized process group!`
- Training will begin automatically once both devices are connected

---

## What You'll See During Training

Once both devices are connected, you'll see:

**On Master Device (Rank 0):**
```
Rank 0: Successfully initialized process group!
Using device: cpu (Forced for CPU Testing)
World size: 2, Rank: 0, Backend: gloo

Epoch 1
Epoch 1, Batch 0/50, Loss: 3.2958, Acc: 3.70%
Epoch 1, Batch 10/50, Loss: 2.1234, Acc: 15.20%
...
Train - Loss: 2.1234, Acc: 15.20%
Val   - Loss: 2.3456, Acc: 12.50%
```

**On Worker Device (Rank 1):**
```
Rank 1: Successfully initialized process group!
Using device: cpu (Forced for CPU Testing)
World size: 2, Rank: 1, Backend: gloo

Epoch 1
(No batch logging - only rank 0 logs)
```

---

## Troubleshooting

### Master device starts but worker can't connect

**Symptoms:**
- Master shows "Waiting for connection..."
- Worker shows connection errors

**Solutions:**
1. Check firewall on master device allows port 29500
2. Verify `MASTER_ADDR` on worker matches master's IP
3. Ensure both devices are on the same network
4. Try pinging master from worker: `ping <master_ip>`

### "Process group timeout" error

**Cause:** Worker device took too long to connect

**Solution:**
- Start worker device within 10 seconds of master
- Or increase timeout in code (not recommended)

### Both devices show errors immediately

**Check:**
- All environment variables are set correctly
- `MASTER_ADDR` is not `0.0.0.0`
- `GLOO_SOCKET_IFNAME` is unset or set to interface name (not IP)
- Virtual environment is activated

---

## Quick Reference Commands

### Master Device (Rank 0)
```powershell
# Set environment variables (if not already set)
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Your master IP
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "0"
$env:LOCAL_RANK = "0"
$env:GLOO_SOCKET_IFNAME = $null

# Activate venv (if not already)
.\venv\Scripts\Activate.ps1

# Start training
python src/train_ddp.py --config experiments/config_jester.yaml
```

### Worker Device (Rank 1)
```powershell
# Set environment variables (if not already set)
$env:USE_LIBUV = "0"
$env:GLOO_SOCKET_FAMILY = "INET"
$env:MASTER_ADDR = "172.20.10.2"  # Master's IP (same as master)
$env:MASTER_PORT = "29500"
$env:WORLD_SIZE = "2"
$env:RANK = "1"
$env:LOCAL_RANK = "0"
$env:GLOO_SOCKET_IFNAME = $null

# Activate venv (if not already)
.\venv\Scripts\Activate.ps1

# Start training (within 5-10 seconds of master)
python src/train_ddp.py --config experiments/config_jester.yaml
```

---

## Stopping Training

To stop training:
- Press `Ctrl+C` on **both devices**
- The master will clean up first, then the worker

**Note:** If you only stop one device, the other will timeout and show errors. This is normal - just stop both.


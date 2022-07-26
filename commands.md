# Development
```
# 1. Rsync repo
cd /Users/pratik/repos/waveglow
watch -d -n5 "rsync -av --exclude-from=\".rsyncignore_upload\" \"/Users/pratik/repos/waveglow\" w:/work/gk77/k77021/repos"

# 2. Rsync data
cd /Users/pratik/data/timbre
rsync -av "/Users/pratik/data/timbre" w:/work/gk77/k77021/data

# 3. checkpoint from wisteria
watch -d -n5 "rsync -av w:/work/gk77/k77021/repos/waveglow/checkpoints \"/Users/pratik/repos/waveglow/checkpointsw\""
rsync -av w:/work/gk77/k77021/repos/waveglow/checkpoints "/Users/pratik/repos/waveglow/checkpointsw"
scp w:/work/gk77/k77021/repos/waveglow/checkpoints/waveglow_4000 /Users/pratik/repos/waveglow/checkpointsw/checkpoints

# logs from wisteria
watch -d -n5 "rsync -av w:/work/gk77/k77021/repos/waveglow/checkpoints/logs \"/Users/pratik/repos/waveglow/checkpointsw/checkpoints\""

```
# Notes

1. Restart tensorboard if it stops showing logs

## Wisteria

```
# for debug jobs 
pjsub wisteria-scripts/wisteria-debug.sh

# for interactive jobs
pjsub --interact wisteria-scripts/wisteria-interactive.sh

```

(base) [k77021@wisteria02 waveglow]$ ls -la checkpoints
total 18828108
drwxrwxr-x 3 k77021 gk77       4096 Jul 21 14:36 .
drwxr-xr-x 6 k77021 gk77       4096 Jul 21 14:35 ..
drwxr-x--- 2 k77021 gk77       4096 Jul 21 14:20 logs
-rw-r----- 1 k77021 gk77 3213321527 Jul 21 14:21 waveglow_0
-rw-r----- 1 k77021 gk77 3213322231 Jul 21 14:36 waveglow_1000
-rw-r----- 1 k77021 gk77 3213321591 Jul 21 14:24 waveglow_200
-rw-r----- 1 k77021 gk77 3213322231 Jul 21 14:22 waveglow_400
-rw-r----- 1 k77021 gk77 3213322231 Jul 21 14:27 waveglow_600
-rw-r----- 1 k77021 gk77 3213322231 Jul 21 14:31 waveglow_800

(base) [k77021@wisteria02 waveglow]$ ls -la checkpoints/logs/
total 104
drwxr-x--- 2 k77021 gk77  4096 Jul 21 14:20 .
drwxrwxr-x 3 k77021 gk77  4096 Jul 21 14:36 ..
-rw-r----- 1 k77021 gk77 20034 Jul 21 14:13 events.out.tfevents.1658380058.wa33
-rw-r----- 1 k77021 gk77 51078 Jul 21 14:35 events.out.tfevents.1658380431.wa35
-rw-r----- 1 k77021 gk77    40 Jul 21 14:17 events.out.tfevents.1658380664.wa33
-rw-r----- 1 k77021 gk77 13274 Jul 21 14:24 events.out.tfevents.1658380856.wa33


run10 - network size

I kept everything from run9 case H (positions only, no target, 1000 points, no
noise, Adam capped at 40,000 iterations) and changed only the network:

  K6_w128:  6 input frequencies,  4 layers x 128   (the default)  646, 821
  K12_w128: 12 input frequencies, 4 layers x 128                  646
  K24_w128: 24 input frequencies, 4 layers x 128                  646, 821
  K12_w256: 12 input frequencies, 4 layers x 256                  646

The network's acceleration is also compared directly against the exact CR3BP
acceleration (accel columns).

Files: arch.py, results.csv

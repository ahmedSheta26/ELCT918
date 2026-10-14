# ELCT918 – Verilog Refresher Assignment: Crack the Cipher Chip

Black-box reverse engineering of `mystery.v` (a gate-level netlist of a small stream-cipher chip), rebuilt as clean RTL and verified cycle for cycle.

## Password

**Plaintext nibbles, one per clock cycle right after reset: `6 3 B 9`**

Encrypted, these give the ciphertext nibbles `C A F E`, which is the secret word, so `flag` goes high after the 4th nibble (`dout = 0x4E`).

## Files

| File | Content |
|---|---|
| `design.sv` | The unchanged `mystery` module, followed by my clean RTL module `actual` (same ports, identical behaviour) |
| `testbench.sv` | Reverse-engineering experiments, the password test, and a self-checking comparison of `mystery` vs `actual` (random stimulus, 40,000+ cycles) |
| `wave.do` | QuestaSim waveform setup (clock, `rst_n`, `en`, `din`, and `dout` / `flag` of both `mystery` and `actual`) |
| `README.md` | This file |

## How to run (QuestaSim)

Run this command in questa terminal in the folder containing the files:

```
do wave.do
```
- The console prints one log line per directed cycle (showing `mystery` and `actual` side by side) and ends with `PASS` or the first mismatches.
- The testbench also writes `dump.vcd`.

## How the chip works (summary)

| Output | Behaviour |
|---|---|
| `dout[7:4]` | Counter of enabled nibbles since reset (4 bits, wraps F → 0) |
| `dout[3:0]` | Ciphertext = plaintext + key (mod 16). The key repeats every 4 enabled nibbles: **6, 7, 4, 5** (= `6 XOR position`) |
| `flag` | High while the last four ciphertext nibbles are `C A F E`. Held while `en = 0` |

On every rising clock edge:

- `rst_n = 0`: the counter and the remembered ciphertext are cleared (reset wins over `en`).
- `rst_n = 1`, `en = 0`: everything holds and `din` is ignored.
- `rst_n = 1`, `en = 1`: `cipher = din + (6 ^ count[1:0])`, the cipher is shifted into a 16-bit history of the last four ciphertext nibbles, and the counter increments.

## How the password was found (no netlist tracing)

1. Experiments showed the lower nibble of `dout` is a keyed ciphertext whose key changes with position (table of all 16 plaintexts × 4 positions).
2. The assignment says `flag` depends on the last four ciphertext nibbles, so I tried all 65,536 four-nibble messages directly on the chip. Exactly one raised `flag`: `6 3 B 9`.
3. Cross-check: subtracting the keys 6, 7, 4, 5 from `CAFE` gives `6 3 B 9`, the same message. Each position is a one-to-one addition, so the answer is unique.

## Self-check result

```
cycles compared : 40005
flag-high cycles: 2065   counter wraps: 1091
PASS: actual matches mystery on every cycle (0 mismatches)
```

The testbench compares `dout` and `flag` on every cycle, covers resets at random times, `en` toggling, long `en = 1` runs (counter wrap), and `CAFE` / near-`CAFE` sequences at every alignment.

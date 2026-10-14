vlog -timescale 1ns/1ps -f source.f
vsim -voptargs="+acc" work.tb

add wave \
/tb/clk \
/tb/rst_n \
/tb/en \
/tb/din \
/tb/dout_m \
/tb/dout_a \
/tb/flag_m \
/tb/flag_a \
/tb/mismatches \
/tb/wraps


# Run the simulation for 1000 ns
run 500 us

# Fit the entire run into the waveform window
wave zoom full
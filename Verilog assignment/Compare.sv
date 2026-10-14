`timescale 1ns/1ps
// =====================================================================
//  Testbench: reverse-engineering experiments + self-checking comparison
//  of the black box (mystery) against the clean RTL (solution).
//  Both modules always receive identical inputs; every cycle dout/flag are compared.
// =====================================================================
module tb;
  reg        clk = 0, rst_n = 0, en = 0;   // shared stimulus
  reg  [3:0] din = 0;
  wire [7:0] dout_m, dout_a;               // outputs of mystery (_m) and solution (_a)
  wire       flag_m, flag_a;

  mystery bb  (.clk(clk), .rst_n(rst_n), .en(en), .din(din), .dout(dout_m), .flag(flag_m)); // black box
  solution  dut (.clk(clk), .rst_n(rst_n), .en(en), .din(din), .dout(dout_a), .flag(flag_a)); // our RTL

  always #5 clk = ~clk;                    // 10 ns clock period

  // ---------------- bookkeeping ----------------
  integer cycle      = 0;                  // clock cycles driven so far
  integer mismatches = 0;                  // cycles where outputs differed
  integer flag_cycles = 0;                 // cycles where the chip's flag was high (coverage)
  integer wraps      = 0;                  // times the counter wrapped F -> 0 (coverage)
  reg     verbose    = 1;                  // print a log line per cycle?
  reg [1:0] pos      = 0;                  // testbench's own count of enabled nibbles (mod 4), only used to build stimulus
  integer seed       = 32'hC0FFEE;         // seed for $random -> reproducible runs

  // ---------------- core task: drive one clock cycle and compare ----------------
  task drive(input r, input e, input [3:0] d);
    begin
      @(negedge clk); rst_n = r; en = e; din = d;       // change inputs away from the active edge
      @(posedge clk); #1;                               // outputs are registered: look just after the edge
      cycle = cycle + 1;
      if (!r) pos = 0; else if (e) pos = pos + 1;       // track message position for stimulus generation
      if (flag_m === 1'b1) flag_cycles = flag_cycles + 1;
      if (r && e && dout_m[7:4] == 4'h0) wraps = wraps + 1; // counter just wrapped to 0
      if (verbose)
        $display("t=%0t rst_n=%b en=%b din=%h -> mystery: dout=%h flag=%b | solution: dout=%h flag=%b",
                 $time, r, e, d, dout_m, flag_m, dout_a, flag_a);
      if (dout_m !== dout_a || flag_m !== flag_a) begin // 4-state compare (catches X too)
        mismatches = mismatches + 1;
        if (mismatches <= 10)                           // only show the first mismatches
          $display("MISMATCH #%0d at cycle %0d (t=%0t): rst_n=%b en=%b din=%h | mystery dout=%h flag=%b | solution dout=%h flag=%b",
                   mismatches, cycle, $time, r, e, d, dout_m, flag_m, dout_a, flag_a);
      end
    end
  endtask

  task step(input e, input [3:0] d);  drive(1'b1, e, d); endtask   // normal cycle (no reset)
  task do_reset;                      drive(1'b0, 1'b0, 4'h0); endtask // one cycle of reset

  // Send one plaintext that makes the ciphertext equal 'target' at the current position.
  // (key = 6 ^ pos, learned from experiment 3)
  task send_cipher(input [3:0] target);
    reg [3:0] key;
    begin key = 4'h6 ^ {2'b00, pos}; step(1'b1, target - key); end
  endtask

  // ---------------- variables for experiments ----------------
  integer i, j, k, n, mode, len;
  reg [3:0] tab [0:3][0:15];               // cipher table: [position mod 4][plaintext]
  reg [15:0] target_word;
  reg [31:0] rnd;

  initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, tb);

    // ===== Experiment 0: reset =====
    $display("\n=== EXP 0: reset (rst_n low for 2 clocks) ===");
    repeat (2) @(posedge clk); #1;
    $display("after reset: mystery dout=%h flag=%b | solution dout=%h flag=%b", dout_m, flag_m, dout_a, flag_a);
    if (dout_m !== dout_a || flag_m !== flag_a) mismatches = mismatches + 1;
    rst_n = 1;

    // ===== Experiment 1: what happens when en is low? =====
    $display("\n=== EXP 1: en = 0 -> chip must ignore din and hold ===");
    step(0, 4'h5); step(0, 4'h9);

    // ===== Experiment 2: same plaintext many times (counter + changing key), incl. wrap-around =====
    $display("\n=== EXP 2: din=1 repeated, then en=0 pause, then en=1 (counter wrap) ===");
    repeat (15) step(1, 4'h1);
    step(0, 4'h1);                         // pause: nothing should change
    step(1, 4'h1);                         // 16th enabled nibble -> counter wraps to 0

    // ===== Experiment 3: full cipher table (every plaintext at every position mod 4) =====
    $display("\n=== EXP 3: cipher table, ciphertext of each plaintext at positions 0..3 ===");
    verbose = 0;
    for (i = 0; i < 16; i = i + 1) begin
      do_reset;
      for (k = 0; k < 4; k = k + 1) begin step(1, i[3:0]); tab[k][i] = dout_m[3:0]; end
    end
    $write("plain : "); for (i = 0; i < 16; i = i + 1) $write("%h ", i[3:0]); $write("\n");
    for (k = 0; k < 4; k = k + 1) begin
      $write("pos %0d : ", k); for (i = 0; i < 16; i = i + 1) $write("%h ", tab[k][i]); $write("   (key = cipher - plain = %h)\n", tab[k][0]);
    end
    verbose = 1;

    // ===== Experiment 4: reset in the middle of a message restarts the key/counter =====
    $display("\n=== EXP 4: reset in the middle of a message ===");
    do_reset; step(1, 4'h3); step(1, 4'h3); step(1, 4'h3);
    drive(1'b0, 1'b1, 4'h3);               // reset with en=1 : reset must win
    step(1, 4'h3);                         // behaves like the first nibble again

    // ===== Task 3: the password =====
    $display("\n=== PASSWORD TEST: plaintext 6 3 B 9 -> ciphertext C A F E -> flag ===");
    do_reset;
    step(1, 4'h6); step(1, 4'h3); step(1, 4'hB); step(1, 4'h9);   // flag must rise on the 4th nibble
    if (flag_m !== 1'b1 || flag_a !== 1'b1) begin $display("PASSWORD FAILED"); mismatches = mismatches + 1; end
    else $display(">>> flag went HIGH on the 4th nibble (dout=%h)", dout_m);
    step(0, 4'h0); step(0, 4'hF);          // en=0 : flag is held
    step(1, 4'h0);                         // any further nibble pushes CAFE out -> flag drops
    $display("--- near miss (last plaintext 8 instead of 9): flag must stay low ---");
    do_reset;
    step(1, 4'h6); step(1, 4'h3); step(1, 4'hB); step(1, 4'h8);
    $display("--- CAFE ciphertext at a different alignment (after 2 junk nibbles) ---");
    do_reset;
    step(1, 4'h0); step(1, 4'h0);
    send_cipher(4'hC); send_cipher(4'hA); send_cipher(4'hF); send_cipher(4'hE);

    // ===== Self-check: long random run (>= 40000 cycles) =====
    $display("\n=== RANDOM TEST: random inputs, en toggling, random resets, rare CAFE sequences ===");
    verbose = 0;
    mode = 0; len = 0;
    while (cycle < 40000) begin
      if (len == 0) begin mode = $unsigned($random(seed)) % 4; len = 20 + $unsigned($random(seed)) % 200; end
      len = len - 1;
      rnd = $random(seed);
      case (mode)
        0: drive(rnd[0]|rnd[1]|rnd[2]|rnd[3]|rnd[4]|rnd[5], rnd[8], rnd[15:12]);      // en 50/50, reset 1/64
        1: drive(rnd[0]|rnd[1]|rnd[2]|rnd[3]|rnd[4]|rnd[5]|rnd[6]|rnd[7], 1'b1, rnd[15:12]); // long en=1 runs -> counter wraps
        2: drive(rnd[0]|rnd[1]|rnd[2]|rnd[3]|rnd[4]|rnd[5], rnd[9]&rnd[8]&rnd[10], rnd[15:12]); // mostly en=0
        3: begin                                                                          // rare sequences: CAFE (or near-CAFE) at random alignment
             target_word = 16'hCAFE;
             if (rnd[17]) target_word[4*rnd[19:18] +: 4] = rnd[23:20];                    // 50%: corrupt one nibble (near miss)
             if (rnd[16]) do_reset;                                                      // sometimes start from a fresh message
             for (n = 0; n < rnd[26:24]; n = n + 1) step(1'b1, rnd[31:28] + n);          // 0..7 junk nibbles -> random alignment
             for (n = 3; n >= 0; n = n - 1) send_cipher(target_word[4*n +: 4]);          // C,A,F,E
             if (rnd[27]) step(rnd[29], rnd[15:12]);                                     // sometimes one more cycle after
           end
      endcase
    end

    // ===== Verdict =====
    $display("\n----------------------------------------------");
    $display("cycles compared : %0d", cycle);
    $display("flag-high cycles: %0d   counter wraps: %0d", flag_cycles, wraps);
    if (mismatches == 0) $display("PASS: solution matches mystery on every cycle (0 mismatches)");
    else                 $display("FAIL: %0d mismatching cycles (first ones listed above)", mismatches);
    $display("----------------------------------------------");
    $finish;
  end
endmodule

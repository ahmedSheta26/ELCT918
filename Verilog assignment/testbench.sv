`timescale 1ns/1ps
module tb;
  reg clk = 0, rst_n = 0, en = 0;
  reg [3:0] din = 0;
  wire [7:0] dout;
  wire flag;
  mystery bb (.clk(clk), .rst_n(rst_n), .en(en), .din(din), .dout(dout), .flag(flag));
  always #5 clk = ~clk;
  // Apply one input for one clock cycle, then print what the chip shows
  task step(input e, input [3:0] d);
    begin
      @(negedge clk); en = e; din = d;
      @(posedge clk); #1;
      $display("t=%0t en=%b din=%b -> dout=%b flag=%b", $time, e, d, dout, flag);
    end
  endtask
  initial begin
    $dumpfile("dump.vcd"); $dumpvars(0, tb);
    repeat (2) @(posedge clk); #1;
    $display("after reset: dout=%h flag=%b", dout, flag);
    rst_n = 1;
    // Experiment 1: what happens when en is low?
    step(0, 4'h5); step(0, 4'h9);
    // Experiment 2: ... your experiments here ...
    repeat (15) step(1, 4'h1);
    step(0, 4'h1);
    step(1, 4'h1);
    $finish;
  end
endmodule
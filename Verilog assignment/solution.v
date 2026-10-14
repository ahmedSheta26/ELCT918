// =====================================================================
//  solution : clean RTL re-implementation of the "mystery" chip
//  Same ports as mystery, same behaviour on every clock cycle.
//
//  What the chip does (found purely by experiments, see testbench):
//   * dout[7:4] : message counter - counts enabled nibbles, wraps 15 -> 0
//   * dout[3:0] : ciphertext = plaintext + key  (mod 16)
//                 key repeats every 4 nibbles: 6,7,4,5  (= 6 XOR position)
//   * the last 4 ciphertext nibbles are remembered
//   * flag      : high while the last 4 ciphertext nibbles are C,A,F,E
// =====================================================================
module solution (clk, rst_n, en, din, dout, flag);
  input        clk;      // rising-edge clock
  input        rst_n;    // synchronous, active-low reset (starts a new message)
  input        en;       // high = encrypt the nibble on din at this clock edge
  input  [3:0] din;      // plaintext nibble
  output [7:0] dout;     // {message counter, latest ciphertext nibble}
  output       flag;     // high when last 4 ciphertext nibbles spell the secret word

  // ---------------- constants ----------------
  localparam [3:0]  BASE_KEY = 4'h6;     // key used for the 1st nibble of a message
  localparam [15:0] SECRET   = 16'hCAFE; // secret word (oldest nibble = C, newest = E)

  // ---------------- state ----------------
  reg [3:0]  count;      // number of nibbles encrypted since reset (mod 16) -> dout[7:4]
  reg [15:0] history;    // last 4 ciphertext nibbles: [3:0]=newest ... [15:12]=oldest

  // ---------------- cipher ----------------
  // The key changes with the nibble position: only the 2 LSBs of count matter,
  // so the key sequence 6,7,4,5 repeats every 4 enabled cycles.
  wire [3:0] key    = BASE_KEY ^ {2'b00, count[1:0]};  // per-position key
  wire [3:0] cipher = din + key;                       // add key to plaintext, wraps mod 16

  // ---------------- sequential logic ----------------
  always @(posedge clk) begin
    if (!rst_n) begin                      // synchronous reset has priority over everything
      count   <= 4'd0;                     // new message: position 0
      history <= 16'h0000;                 // forget previous ciphertext
    end else if (en) begin                 // only enabled cycles encrypt
      count   <= count + 4'd1;             // next nibble position (wraps 15 -> 0)
      history <= {history[11:0], cipher};  // shift in newest ciphertext nibble
    end
    // en = 0 : everything holds its value
  end

  // ---------------- outputs ----------------
  assign dout = {count, history[3:0]};     // counter in the upper nibble, newest ciphertext in the lower
  assign flag = (history == SECRET);       // high when the last 4 ciphertexts == C,A,F,E (registered state -> held while en=0)
endmodule

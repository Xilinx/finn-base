module incomplete_table (
	input [15:0] in,
	 output reg result
);

	always @(in) begin
		case(in)
			16'b0000000000000001 : result = 1'b1;
			16'b0000000000111010 : result = 1'b1;
			16'b0000000000001111 : result = 1'b1;
			16'b0000000001011001 : result = 1'b1;
			16'b0000001010110111 : result = 1'b1;
			16'b0001100101010101 : result = 1'b1;
			default: result = 1'b0;
		endcase
	end
endmodule

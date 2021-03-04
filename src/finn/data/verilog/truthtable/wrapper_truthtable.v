
`timescale 1 ns / 1 ps

(* CORE_GENERATION_INFO="wrapper_truthtable,hls_ip_2019_1,{HLS_INPUT_TYPE=cxx,HLS_INPUT_FLOAT=0,HLS_INPUT_FIXED=1,HLS_INPUT_PART=xc7z020-clg400-1,HLS_INPUT_CLOCK=5.000000,HLS_INPUT_ARCH=others,HLS_SYN_CLOCK=3.552000,HLS_SYN_LAT=0,HLS_SYN_TPT=none,HLS_SYN_MEM=0,HLS_SYN_DSP=0,HLS_SYN_FF=144,HLS_SYN_LUT=271,HLS_VERSION=2019_1}" *)


module wrapper_truthtable (
    input_data,
    result_data
);

input [15:0]input_data;
 output result_data;

incomplete_table my_incomplete_table(
    .in(input_data),
    .result(result_data)
);

endmodule //wrapper_truthtable

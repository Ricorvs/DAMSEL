set terminal pdf noenhanced
set key reverse above Left vertical maxrows 1
set palette rgb 33,13,10

set yrange[0:]
set style fill solid 0.2

warpfiles = system("find ".folder."/*/warps.csv")
kerneltimes = system("find ".folder."/*/kerneltimes.csv")
blockfiles = system("find ".folder."/*/blocks.csv")
dirs = system("find ".folder."/*/ -type d")

set output "warps_".outputfile
set xlabel "Parallel Warps"
set ylabel "Throughput (TOPS)"
do for [i=9:0:-3] {
    plot folder."/".(i)."/warps.csv" u 2:($5 / 1e12) w lines title "16x16x16",\
         folder."/".(i + 1)."/warps.csv" u 2:($5 / 1e12) w lines title "32x8x16",\
         folder."/".(i + 2)."/warps.csv" u 2:($5 / 1e12) w lines title "8x32x16"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}
do for [i=0:2] {
    plot folder."/".(i)."/warps.csv" u 2:($5 / 1e12) w lines title "FP16, FP32",\
         folder."/".(i + 3)."/warps.csv" u 2:($5 / 1e12) w lines title "FP16, FP16",\
         folder."/".(i + 6)."/warps.csv" u 2:($5 / 1e12) w lines title "INT8, INT32"
}


set output "warps1_".outputfile
set xlabel "Parallel Warps"
set ylabel "Throughput (TOPS)"
set yrange[0:*]
do for [i=9:0:-1] {
    plot folder."/".(i)."/warps1.csv" every 1::::31 u 2:($5 / 1e12) w lines title "1 Block",\
         folder."/".(i)."/warps1.csv" every 1::32 u 2:($5 / 1e12) w lines title "2 Blocks"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}


unset yrange
set yrange[0:]
set output "kerneltimes_".outputfile
set xlabel "Blocks per SM"
set ylabel "Throughput (TOPS)"
do for [i=9:0:-3] {
    plot folder."/".(i)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "16x16x16",\
         folder."/".(i + 1)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "32x8x16",\
         folder."/".(i + 2)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "8x32x16"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}
do for [i=0:2] {
    plot folder."/".(i)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "FP16, FP32",\
         folder."/".(i + 3)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "FP16, FP16",\
         folder."/".(i + 6)."/kerneltimes.csv" u 3:($5 / 1e12) w lines title "INT8, INT32"
}
unset yrange
set yrange[0:]
set output "blocks_".outputfile
set xlabel "Blocks Started"
do for [i=9:0:-3] {
    plot folder."/".(i)."/blocks.csv" u 3:($5 / 1e12) w lines title "16x16x16",\
         folder."/".(i + 1)."/blocks.csv" u 3:($5 / 1e12) w lines title "32x8x16",\
         folder."/".(i + 2)."/blocks.csv" u 3:($5 / 1e12) w lines title "8x32x16"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}
do for [i=0:2] {
    plot folder."/".(i)."/blocks.csv" u 3:($5 / 1e12) w lines title "FP16, FP32",\
         folder."/".(i + 3)."/blocks.csv" u 3:($5 / 1e12) w lines title "FP16, FP16",\
         folder."/".(i + 6)."/blocks.csv" u 3:($5 / 1e12) w lines title "INT8, INT32"
}
unset yrange
set yrange[0:]
set output "blockstimes_".outputfile
set xlabel "Blocks Started"
set ylabel "Execution time (s)"
do for [i=0:9:3] {
    plot folder."/".(i)."/blocks.csv" u 3:($4) w lines title "16x16x16",\
         folder."/".(i + 1)."/blocks.csv" u 3:($4) w lines title "32x8x16",\
         folder."/".(i + 2)."/blocks.csv" u 3:($4) w lines title "8x32x16"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}
do for [i=0:2] {
    plot folder."/".(i)."/blocks.csv" u 3:($4) w lines title "FP16, FP32",\
         folder."/".(i + 3)."/blocks.csv" u 3:($4) w lines title "FP16, FP16",\
         folder."/".(i + 6)."/blocks.csv" u 3:($4) w lines title "INT8, INT32"
}

unset yrange
set yrange[0:*]
set style fill solid 1
set cblabel "Block ID"
set print "-"
do for [b=2:5] {
    blocks = 2**b
    set cbrange[0:4 * 32 / blocks - 1]
    set cbtics ceil(16.0 / blocks)
    do for [i=0:11] {
        files = system("find ".folder."/".i."/kerneltimes/*".blocks.".csv -not -type d | sort")
        set output folder."/".i."/warptimes_".blocks.".pdf"
        set xlabel "Warp ID"
        set ylabel "Elapsed Time (Gigacycles)"
        set ytics
        do for [j=1:words(files)] {
            plot word(files, j) u 0:(($3 - $2) / 1e9):(floor($0 / blocks)) lt 7 ps 0.3 lc palette notitle
        }

        set output folder."/".i."/warpdurations_".blocks.".pdf"
        unset ylabel
        unset ytics
        set xlabel "Elapsed Time (Gigacycles)"
        do for [j=1:words(files)] {
            plot word(files, j) u ($2 / 1e9):0:(($3 - $2) / 1e9):(0):(floor($0 / blocks)) w vectors lc palette nohead notitle
        }
    }

}
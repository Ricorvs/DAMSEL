set terminal pdf noenhanced
set key reverse above Left vertical maxrows 1 width -1
set palette rgb 33,13,10

configs = "19 17 18 20 21"
set yrange[0:1.4]
set style fill solid 0.2

set output "warps_".outputfile
set xlabel "Parallel Warps"
set ylabel "Throughput (TOPS)"
plot folder."/0_20/warps1_0.csv" u 2:($5 * 16 / 1e12) w lines lc 7 title "FP64", \
     folder."/0_21/warps1_0.csv" u 2:($5 * 16 / 1e12) w lines lc 4 title "FP32",\
     folder."/0_17/warps1_0.csv" u 2:($5 * 16 / 1e12) w lines lc 1 title "FP16",\
     folder."/0_18/warps1_0.csv" u 2:($5 * 16 / 1e12) w lines lc 2 title "FP16x2",\
     folder."/0_19/warps1_0.csv" u 2:($5 * 16 / 1e12) w lines lc 3 title "INT32"

set yrange[0:*]
set output "warps1_".outputfile
set xlabel "Parallel Warps"
set ylabel "Throughput (TOPS)"
plot folder."/0_20/warps1_1.csv" u 2:($5 * 16 / 1e12) w lines lc 7 title "FP64", \
     folder."/0_21/warps1_1.csv" u 2:($5 * 16 / 1e12) w lines lc 4 title "FP32",\
     folder."/0_17/warps1_1.csv" u 2:($5 * 16 / 1e12) w lines lc 1 title "FP16",\
     folder."/0_18/warps1_1.csv" u 2:($5 * 16 / 1e12) w lines lc 2 title "FP16x2",\
     folder."/0_19/warps1_1.csv" u 2:($5 * 16 / 1e12) w lines lc 3 title "INT32"


unset yrange
set yrange[0:90]
set output "blocks_".outputfile
set xlabel "Blocks Started"
plot folder."/0_20/blocks.csv" u 3:($5 * 16 / 1e12) w lines lc 7 title "FP64", \
     folder."/0_21/blocks.csv" u 3:($5 * 16 / 1e12) w lines lc 4 title "FP32",\
     folder."/0_17/blocks.csv" u 3:($5 * 16 / 1e12) w lines lc 1 title "FP16",\
     folder."/0_18/blocks.csv" u 3:($5 * 16 / 1e12) w lines lc 2 title "FP16x2",\
     folder."/0_19/blocks.csv" u 3:($5 * 16 / 1e12) w lines lc 3 title "INT32"

set output "blocksdiffs_".outputfile
set yrange[0:*]
do for [k=1:words(configs)] {
    i = word(configs, k)
    plot folder."/0_".(i)."/blocks.csv" u 3:($5 * 16 / 1e12) w lines title "Default fma",\
         folder."/1_".(i)."/blocks.csv" u 3:($5 * 16 / 1e12) w lines title "Timing fma", \
         folder."/2_".(i)."/blocks.csv" u 3:($5 * 16 / 1e12) w lines title "Default",\
         folder."/3_".(i)."/blocks.csv" u 3:($5 * 16 / 1e12) w lines title "Timing"
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}
unset yrange
set yrange[0:]
set output "blockstimes_".outputfile
set xlabel "Blocks Started"
set ylabel "Execution time (s)"
plot folder."/0_20/blocks.csv" u 3:($4) w lines lc 7 title "FP64", \
     folder."/0_21/blocks.csv" u 3:($4) w lines lc 4 title "FP32",\
     folder."/0_17/blocks.csv" u 3:($4) w lines lc 1 title "FP16",\
     folder."/0_18/blocks.csv" u 3:($4) w lines lc 2 title "FP16x2",\
     folder."/0_19/blocks.csv" u 3:($4) w lines lc 3 title "INT32"

unset yrange
set yrange[0:*]
set style fill solid 1
set cblabel "Block ID"
set print "-"
do for [b=2:5] {
    blocks = 2**b
    set cbrange[0:4 * 32 / blocks - 1]
    set cbtics ceil(16.0 / blocks)
    do for [k=1:words(configs)] {
        set yrange[0:3]
        i = word(configs, k)
        files = system("find ".folder."/1_".i."/kerneltimes/*".blocks.".csv -not -type d | sort")
        set output folder."/1_".i."/warptimes_".blocks.".pdf"
        set xlabel "Warp ID"
        set ylabel "Elapsed Time (Gigacycles)"
        set ytics
        do for [j=1:words(files)] {
            plot word(files, j) u 0:(($3 - $2) / 1e9):(floor($0 / blocks)) lt 7 ps 0.3 lc palette notitle
        }
        set yrange[0:*]

        set output folder."/1_".i."/warpdurations_".blocks.".pdf"
        unset ylabel
        unset ytics
        set xlabel "Elapsed Time (Gigacycles)"
        do for [j=1:words(files)] {
            plot word(files, j) u ($2 / 1e9):0:(($3 - $2) / 1e9):(0):(floor($0 / blocks)) w vectors lc palette nohead notitle
        }
    }

}
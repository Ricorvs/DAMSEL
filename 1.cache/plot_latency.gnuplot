set terminal pdf noenhanced font ",16"
set key reverse above Left vertical maxrows 1
set yrange[0:]
set output "cache_latency_diff_1080ti_2080ti_A100.pdf"

# set output "A100.pdf"
set xlabel "Array size (KiB)"
set ylabel "Average access time (cycles)"

set xrange[23.5:24.5]
plot "results/1080ti/ca/default/preheat1_stride1.csv" u ($1 / 1024):4 w lines title "GTX 1080Ti"

set output "single_run_latency_".outputfile
set xlabel "Access offset (B)"
set ylabel "Access time (cycles)"

# set key font ",16"
set xrange[0:512]
set yrange[0:400]

plot folder."/ca/single_runs/0.csv" u 2:3 pt 7 ps 0.3 lc rgb "red" title "Stride 4B", \
     folder."/ca/single_runs/2.csv" u 2:3 pt 7 ps 0.3 lc rgb "blue" title "Stride 16B", \
     folder."/ca/single_runs/3.csv" u 2:3 pt 7 ps 0.3 lc rgb "green" title "Stride 32B"


unset xrange
set xrange[0:60000]
unset yrange
set yrange[0:]
set output "single_sm_impact_".outputfile
set palette rgb 33,13,10
set offsets graph 0, 0, 0.05, 0.05
set xlabel "Elapsed time (Cycles)"
set ylabel "Access offset (KiB)"
set cblabel "Warp ID"
set cbrange [0:32]
do for [j=32:1:-1] {
    plot for [i=0:j - 1] folder."/ca/single_sm/".j."_warps.csv" u i * 2 + 3:($2 * 4 / 1024):(column(i * 2 + 4) - column(i * 2 + 3)):(0):(i) w vectors lc palette nohead notitle
    set xrange[GPVAL_X_MIN:GPVAL_X_MAX]
}

set ylabel "Access time (Cycles)"
set xlabel "Access offset (KiB)"
set xrange[0:*]
set yrange[0:*]
unset cblabel
unset cbrange
set output "multi_sm_impact_".outputfile
do for [j=4:1:-1] {
    plot for [i=0:j - 1] folder."/ca/multi_sm/".j."_warps.csv" u ($2 * 4 / 1024):(column(i * 2 + 4) - column(i * 2 + 3)):(i) pt 7 ps 0.1 notitle
    set yrange[GPVAL_Y_MIN:GPVAL_Y_MAX]
}

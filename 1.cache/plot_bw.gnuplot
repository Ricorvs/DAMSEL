set terminal pdf noenhanced 
set key reverse above Left vertical maxrows 1

set xrange[0:*]
set yrange[-1:*]
set cbrange[0:4]
set output "bw_scheduling_".outputfile
set xlabel "Elapsed time (cycles)"

unset ylabel
unset ytics
set cblabel "Block ID"
set cbtics 0,1,4
files = system("find ".folder."/ca/bw/32/NEG/bw_* -not -type d")
set palette rgb 33,13,10

do for [i=1:words(files)] {
    plot word(files, i) u 4:0:($5 - $4):(0):($2) w vectors lc palette nohead notitle
}

set font ",16"
set style fill solid 0.2

set output "bw_diff_32_".outputfile
box = 0.2
set xrange[-0.5:2.5]
unset yrange
set ylabel "Bandwidth (Bytes/Cycle)"
set xlabel "Started Blocks"
set yrange[0:128]
set xtics ("1" 0, "2" 1, "4" 2)
unset ytics
set ytics
set offsets graph 0, 0, 0, 0
do for [i=0:5] {
    plot folder."/ca/bw/32/ADD/bw.csv" every 6::i::18 u ($0 - 1 * box):6:(box) w boxes title "Addition", \
         folder."/ca/bw/32/NEG/bw.csv" every 6::i::18 u ($0 - 0 * box):6:(box) w boxes title "Negation", \
         folder."/ca/bw/32/MOV/bw.csv" every 6::i::18 u ($0 + 1 * box):6:(box) w boxes title "Move"
}

set output "bw_diff_64_".outputfile
do for [i=0:5] {
    plot folder."/ca/bw/64/ADD/bw.csv" every 6::i::18 u ($0 - 1 * box):6:(box) w boxes title "Addition", \
         folder."/ca/bw/64/NEG/bw.csv" every 6::i::18 u ($0 - 0 * box):6:(box) w boxes title "Negation", \
         folder."/ca/bw/64/MOV/bw.csv" every 6::i::18 u ($0 + 1 * box):6:(box) w boxes title "Move"
}

set key reverse above Left vertical maxrows 2
set output "bw_diff_".outputfile
do for [i=0:5] {
    plot folder."/ca/bw/32/ADD/bw.csv" every 6::i::18 u ($0 - 1.5 * box):6:(box) w boxes title "32b Addition", \
         folder."/ca/bw/64/ADD/bw.csv" every 6::i::18 u ($0 + 0.5 * box):6:(box) w boxes title "64b Addition", \
         folder."/ca/bw/32/NEG/bw.csv" every 6::i::18 u ($0 - 0.5 * box):6:(box) w boxes title "32b Negation", \
         folder."/ca/bw/64/NEG/bw.csv" every 6::i::18 u ($0 + 1.5 * box):6:(box) w boxes title "64b Negation", \
}
set terminal pdf noenhanced
set key reverse above Left vertical

set xrange[0:*]
set yrange[0:*]

set xlabel "Blocks Started"
set ylabel "Throughput (GiB/s)"
set key reverse above left Left vertical maxrows 2 width -1 font ",11"
file = folder."/blocks/ca_wb"
rows = system("cat ".folder."/blocks/ca_wb/res_1.csv | wc -l")
set output "blocks_".outputfile
set style fill solid 1

plot folder."/blocks/ca_wb/res_1.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lc 2 notitle, \
     folder."/blocks/cg_wb/res_1.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lt 3 dt 4 lc 2 notitle, \
     folder."/blocks/ca_wb/res_2.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lc 3 notitle, \
     folder."/blocks/cg_wb/res_2.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lt 3 dt 4 lc 3 notitle, \
     folder."/blocks/ca_wb/res_3.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lc 4 notitle, \
     folder."/blocks/cg_wb/res_3.csv" every 1::::(rows / 2 - 1) u ($4):($7 / 1024**3) w lines lt 3 dt 4 lc 4 notitle, \
     NaN w boxes lc 2 title "Row-major read", \
     NaN w lines lc rgb "gray" title "ca", \
     NaN w boxes lc 3 title "Column-major read", \
     NaN w lines lt 3 dt 4 lc rgb "gray" title "cg", \
     NaN w boxes lc 4 title "Cache-optimised"

set xrange[4995:20000]
set output "matrix_size_".outputfile
plot folder."/mat_size/ca_wb/res_1.csv" every 1 u ($1):($7 / 1024**3) w lines lc 2 notitle, \
     folder."/mat_size/cg_wb/res_1.csv" every 1 u ($1):($7 / 1024**3) w lines lt 3 dt 4 lc 2 notitle, \
     folder."/mat_size/ca_wb/res_2.csv" every 1 u ($1):($7 / 1024**3) w lines lc 3 notitle, \
     folder."/mat_size/cg_wb/res_2.csv" every 1 u ($1):($7 / 1024**3) w lines lt 3 dt 4 lc 3 notitle, \
     folder."/mat_size/ca_wb/res_3.csv" every 1 u ($1):($7 / 1024**3) w lines lc 4 notitle, \
     folder."/mat_size/cg_wb/res_3.csv" every 1 u ($1):($7 / 1024**3) w lines lt 3 dt 4 lc 4 notitle, \
     NaN w boxes lc 2 title "Row-major read", \
     NaN w lines lc rgb "gray" title "ca", \
     NaN w boxes lc 3 title "Column-major read", \
     NaN w lines lt 3 dt 4 lc rgb "gray" title "cg", \
     NaN w boxes lc 4 title "Cache-optimised"

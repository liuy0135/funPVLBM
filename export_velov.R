# Obtain the original public dataset and export the complete Monday-Sunday week.
# Run from the repository root: Rscript export_velov.R
archive <- tempfile(fileext='.tar.gz')
folder <- tempfile()
dir.create(folder)
download.file('https://cran.r-project.org/src/contrib/funFEM_1.2.tar.gz', archive, mode='wb')
untar(archive, files='funFEM/data/velov.rda', exdir=folder)
e <- new.env()
load(file.path(folder, 'funFEM/data/velov.rda'), envir=e)
x <- e$velov
stopifnot(identical(dim(x$data), c(346L,181L)))
week <- x$data[,14:181]
stopifnot(identical(as.character(x$dates[seq(14,181,by=24)]),
                    c('Lun-00','Mar-00','Mer-00','Jeu-00','Ven-00','Sam-00','Dim-00')))
y <- matrix(as.vector(t(week)), ncol=24, byrow=TRUE)
colnames(y) <- paste0('V',1:24)
write.csv(y, 'velov_week.csv', row.names=FALSE)
cat('Saved velov_week.csv: 346 stations x 7 days, with 24 hourly columns.\n')

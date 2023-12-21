from stelar_spatiotemporal.lib import get_filesystem


if __name__ == "__main__":
    DATADIR = "/home/jens/ownCloud/Documents/3.Werk/0.TUe_Research/0.STELAR/0.VISTA/VISTA_workbench/data/pipeline_example"
    fs = get_filesystem(DATADIR)

    files = fs.glob(DATADIR + "/**/*.RAS")

    for file in files:
        with fs.open(file, "rb") as f:
            print(f.read(10))


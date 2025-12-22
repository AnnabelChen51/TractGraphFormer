#python downsampleFibers.py -inFile 100206.vtp -outFile 100206-td.vtp --outstep 2
import vtk
import logging
import numpy as np
import whitematteranalysis as wma
import argparse

def computeStepSize(inpd, minpts=5):
    """
    Estimate step size between consecutive points along fiber.
    """

    # measure step size (using first fiber (polyline) that has >=5 points)
    # minpts threshold is for robust estimation if polydata has some
    # very short lines.
    cell_idx = 0
    ptids = vtk.vtkIdList()
    inpoints = inpd.GetPoints()
    inpd.GetLines().InitTraversal()
    while (ptids.GetNumberOfIds() < minpts) & (cell_idx < inpd.GetNumberOfLines()):
        inpd.GetLines().GetNextCell(ptids)
        ##    inpd.GetLines().GetCell(cell_idx, ptids)
        ## the GetCell function is not wrapped in Canopy python-vtk
        cell_idx += 1
    # make sure we have some points along this fiber
    # In case all fibers in the brain are really short, treat it the same as no fibers.
    if ptids.GetNumberOfIds() < 5:
        return 0

    # Use points from the middle of the fiber to estimate step length.
    # This is because the step size may vary near endpoints (in order to include
    # endpoints when downsampling the fiber to reduce file size).
    step_size = 0.0
    count = 0.0
    for ptidx in range(1, ptids.GetNumberOfIds()-1):
        point0 = inpoints.GetPoint(ptids.GetId(ptidx))
        point1 = inpoints.GetPoint(ptids.GetId(ptidx + 1))
        step_size += np.sqrt(np.sum(np.power(np.subtract(point0, point1), 2)))
        count += 1
    step_size = step_size / count
    return step_size

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use DCEC for clustering')
    parser.add_argument(
        '-inFile', action="store", dest="inputFile", default='data', help='A file of features.')
    parser.add_argument(
        '-outFile', action="store", dest="outputFile", default="try",
        help='Output folder of clustering results.')
    parser.add_argument('--outstep', default=1, type=float, help='output step size')
    parser.add_argument('--outpercent', default=100, type=float, help='percentage of output fibers')
    parser.add_argument('--outminpts', default=3, type=float, help='output min points')
    parser.add_argument('--outminlen', default=0.01, type=float, help='output min length')
    parser.add_argument('--outmaxlen', default=200, type=float, help='output max length')

    args = parser.parse_args()

    """
    Remove points from inpd to create outpd with step size approximately outstep.
    Output step size will be the greatest multiple of input fiber
    step size that is less than outstep. So if input step size is 1 and output
    requested step size is 2.5, then final output step size will be 2.
    All endpoints are retained.
    """

    inpd=wma.io.read_polydata(args.inputFile)
    outFile = args.outputFile
    outstep=args.outstep
    outpercent=args.outpercent
    outminpts=args.outminpts
    outminlen=args.outminlen
    outmaxlen=args.outmaxlen

    outpd = vtk.vtkPolyData()

    # compute input step size
    instep = computeStepSize(inpd,minpts=outminpts)
    logging.info('Input Step Size:')
    logging.info(instep)

    # keep every nth point
    n = np.floor(outstep / instep)
    # No interpolation of points: if outstep is smaller than
    # instep keep all points
    if outstep < instep:
        n = 1

    # keep a random sample of outpercent of fibers
    # all possible fiber indices
    findices = list(range(0, inpd.GetNumberOfLines()))
    # now find the size of the desired subset of these indices
    outpercent = np.divide(outpercent, 100.0)
    nkeep = int(np.multiply(inpd.GetNumberOfLines(), outpercent))
    # randomly permute instead of selecting, so that below we can
    # try to keep say the desired random 50% of fibers even though
    # we remove some based on length or number of points
    findices = np.random.permutation(findices)
    # findices = np.random.choice(findices, nkeep, replace=False)
    # sort this so the order corresponds to the original in case desired
    # findices = np.sort(findices)

    # figure out minimum number of points to keep
    # two input variables affect this as users may want to set this
    # in number of points, for sanity check, or in mm to actually
    # modify tract according to anatomical length knowledge
    minpts = int(np.floor(np.maximum(outminpts, np.divide(outminlen, instep))))
    logging.info('Minimum points to retain fiber:')
    logging.info(minpts)
    # figure out maximum number of points to keep
    maxpts = int(np.divide(outmaxlen, instep))
    logging.info('Maximum points to retain fiber:')
    logging.info(maxpts)

    # output and temporary objects
    outlines = vtk.vtkCellArray()
    outlines.InitTraversal()
    outpoints = vtk.vtkPoints()
    ptids = vtk.vtkIdList()

    # loop over all lines, inserting into output only
    # the lines (fibers) and points to be kept
    kept = 0
    for lidx in findices:
        # here we assume this is a fiber bundle in Slicer and only has polylines
        inpd.GetCellPoints(lidx, ptids)
        num_points = ptids.GetNumberOfIds()
        # if we keep this line (fiber)
        if (num_points >= minpts) & (num_points <= maxpts):
            # insert kept points into point array and into line array
            keptptids = vtk.vtkIdList()
            for pidx in range(0, num_points):
                # we keep this point if it is the nth or the endpoint
                if (np.mod(pidx, n) == 0) | (pidx == num_points - 1):
                    point = inpd.GetPoints().GetPoint(ptids.GetId(pidx))
                    idx = outpoints.InsertNextPoint(point)
                    keptptids.InsertNextId(idx)
            outlines.InsertNextCell(keptptids)
            kept = kept + 1
            # If we have reached the desired percentage of fibers stop adding more.
            if kept == nkeep:
                break

    # put data into output polydata
    outpd.SetLines(outlines)
    outpd.SetPoints(outpoints)
    wma.io.write_polydata(outpd,outFile)



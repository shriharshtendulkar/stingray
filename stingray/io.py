import logging
import math
import copy
import os
import pickle
import warnings
from collections.abc import Iterable

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.logger import AstropyUserWarning

import stingray.utils as utils

from .utils import assign_value_if_none, is_string, order_list_of_arrays
from .gti import get_gti_from_all_extensions, load_gtis

# Python 3
import pickle

_H5PY_INSTALLED = True

try:
    import h5py
except ImportError:
    _H5PY_INSTALLED = False


def rough_calibration(pis, mission):
    """Make a rough conversion betwenn PI channel and energy.

    Only works for NICER, NuSTAR, and XMM.

    Parameters
    ----------
    pis: float or array of floats
        PI channels in data
    mission: str
        Mission name

    Returns
    -------
    energies : float or array of floats
        Energy values

    Examples
    --------
    >>> rough_calibration(0, 'nustar')
    1.6
    >>> # It's case-insensitive
    >>> rough_calibration(1200, 'XMm')
    1.2
    >>> rough_calibration(10, 'asDf')
    Traceback (most recent call last):
        ...
    ValueError: Mission asdf not recognized
    >>> rough_calibration(100, 'nicer')
    1.0
    """
    if mission.lower() == "nustar":
        return pis * 0.04 + 1.6
    elif mission.lower() == "xmm":
        return pis * 0.001
    elif mission.lower() == "nicer":
        return pis * 0.01
    raise ValueError(f"Mission {mission.lower()} not recognized")


def get_file_extension(fname):
    """Get the extension from the file name.

    If g-zipped, add '.gz' to extension.

    Examples
    --------
    >>> get_file_extension('ciao.tar')
    '.tar'
    >>> get_file_extension('ciao.tar.gz')
    '.tar.gz'
    >>> get_file_extension('ciao.evt.gz')
    '.evt.gz'
    >>> get_file_extension('ciao.a.tutti.evt.gz')
    '.evt.gz'
    """
    fname_root = fname.replace('.gz', '')
    fname_root = os.path.splitext(fname_root)[0]

    return fname.replace(fname_root, '')


def high_precision_keyword_read(hdr, keyword):
    """Read FITS header keywords, also if split in two.

    In the case where the keyword is split in two, like

        MJDREF = MJDREFI + MJDREFF

    in some missions, this function returns the summed value. Otherwise, the
    content of the single keyword

    Parameters
    ----------
    hdr : dict_like
        The FITS header structure, or a dictionary

    keyword : str
        The key to read in the header

    Returns
    -------
    value : long double
        The value of the key, or ``None`` if something went wrong

    """
    try:
        value = np.longdouble(hdr[keyword])
        return value
    except KeyError:
        pass
    try:
        if len(keyword) == 8:
            keyword = keyword[:7]
        value = np.longdouble(hdr[keyword + 'I'])
        value += np.longdouble(hdr[keyword + 'F'])
        return value
    except KeyError:
        return None


def _patch_mission_info(info, mission=None):
    """Add some information that is surely missing in xselect.mdb.

    Examples
    --------
    >>> info = {'gti': 'STDGTI'}
    >>> new_info = _patch_mission_info(info, mission=None)
    >>> new_info['gti'] == info['gti']
    True
    >>> new_info = _patch_mission_info(info, mission="xmm")
    >>> new_info['gti']
    'STDGTI,GTI0'
    """
    if mission is None:
        return info
    if mission.lower() == "xmm" and "gti" in info:
        info["gti"] += ",GTI0"
    return info


def read_mission_info(mission=None):
    """Search the relevant information about a mission in xselect.mdb."""
    curdir = os.path.abspath(os.path.dirname(__file__))
    fname = os.path.join(curdir, "datasets", "xselect.mdb")

    # If HEADAS is defined, search for the most up-to-date version of the
    # mission database
    if os.getenv("HEADAS"):
        hea_fname = os.path.join(os.getenv("HEADAS"), "bin", "xselect.mdb")
        if os.path.exists(hea_fname):
            fname = hea_fname
    if mission is not None:
        mission = mission.lower()

    db = {}
    with open(fname) as fobj:
        for line in fobj.readlines():
            line = line.strip()
            if mission is not None and not line.lower().startswith(mission):
                continue
            if line.startswith("!") or line == "":
                continue
            allvals = line.split()
            string = allvals[0]
            value = allvals[1:]
            if len(value) == 1:
                value = value[0]

            data = string.split(":")[:]
            if mission is None:
                if data[0] not in db:
                    db[data[0]] = {}
                previous_db_step = db[data[0]]
            else:
                previous_db_step = db
            data = data[1:]
            for key in data[:-1]:
                if key not in previous_db_step:
                    previous_db_step[key] = {}
                previous_db_step = previous_db_step[key]
            previous_db_step[data[-1]] = value
    return _patch_mission_info(db, mission)


def _case_insensitive_search_in_list(string, list_of_strings):
    """Search for a string in a list of strings, in a case-insensitive way.

    Example
    -------
    >>> _case_insensitive_search_in_list("a", ["A", "b"])
    'A'
    >>> _case_insensitive_search_in_list("a", ["c", "b"]) is None
    True
    """
    for s in list_of_strings:
        if string.lower() == s.lower():
            return s
    return None


def _get_additional_data(lctable, additional_columns):
    """Get additional data from a FITS data table.

    Parameters
    ----------
    lctable: `astropy.io.fits.fitsrec.FITS_rec`
        Data table
    additional_columns: list of str
        List of column names to retrieve from the table

    Returns
    -------
    additional_data: dict
        Dictionary associating to each additional column the content of the
        table.
    """
    additional_data = {}
    if additional_columns is not None:
        for a in additional_columns:
            key = _case_insensitive_search_in_list(a, lctable._coldefs.names)
            if key is not None:
                additional_data[a] = np.array(lctable.field(key))
            else:
                warnings.warn('Column ' + a + ' not found')
                additional_data[a] = np.zeros(len(lctable))

    return additional_data


def get_key_from_mission_info(info, key, default, inst=None, mode=None):
    """Get the name of a header key or table column from the mission database.

    Many entries in the mission database have default values that can be
    altered for specific instruments or observing modes. Here, if there is a
    definition for a given instrument and mode, we take that, otherwise we use
    the default).

    Parameters
    ----------
    info : dict
        Nested dictionary containing all the information for a given mission.
        It can be nested, e.g. contain some info for a given instrument, and
        for each observing mode of that instrument.
    key : str
        The key to read from the info dictionary
    default : object
        The default value. It can be of any type, depending on the expected
        type for the entry.

    Other parameters
    ----------------
    inst : str
        Instrument
    mode : str
        Observing mode

    Returns
    -------
    retval : object
        The wanted entry from the info dictionary

    Examples
    --------
    >>> info = {'ecol': 'PI', "A": {"ecol": "BLA"}, "C": {"M1": {"ecol": "X"}}}
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="A", mode=None)
    'BLA'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="B", mode=None)
    'PI'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="A", mode="M1")
    'BLA'
    >>> get_key_from_mission_info(info, "ecol", "BU", inst="C", mode="M1")
    'X'
    >>> get_key_from_mission_info(info, "ghghg", "BU", inst="C", mode="M1")
    'BU'
    """
    filt_info = copy.deepcopy(info)
    if inst is not None and inst in filt_info:
        filt_info.update(info[inst])
        filt_info.pop(inst)
    if mode is not None and mode in filt_info:
        filt_info.update(info[inst][mode])
        filt_info.pop(mode)

    if key in filt_info:
        return filt_info[key]
    return default


def lcurve_from_fits(
    fits_file,
    gtistring="GTI",
    timecolumn="TIME",
    ratecolumn=None,
    ratehdu=1,
    fracexp_limit=0.9,
    outfile=None,
    noclobber=False,
    outdir=None,
):
    """Load a lightcurve from a fits file.

    .. note ::
        FITS light curve handling is still under testing.
        Absolute times might be incorrect depending on the light curve format.

    Parameters
    ----------
    fits_file : str
        File name of the input light curve in FITS format

    Returns
    -------
    data : dict
        Dictionary containing all information needed to create a
        :class:`stingray.Lightcurve` object

    Other Parameters
    ----------------
    gtistring : str
        Name of the GTI extension in the FITS file
    timecolumn : str
        Name of the column containing times in the FITS file
    ratecolumn : str
        Name of the column containing rates in the FITS file
    ratehdu : str or int
        Name or index of the FITS extension containing the light curve
    fracexp_limit : float
        Minimum exposure fraction allowed
    noclobber : bool
        If True, do not overwrite existing files
    """
    warnings.warn(
        """WARNING! FITS light curve handling is still under testing.
        Absolute times might be incorrect."""
    )
    # TODO:
    # treat consistently TDB, UTC, TAI, etc. This requires some documentation
    # reading. For now, we assume TDB
    from astropy.io import fits as pf
    from astropy.time import Time
    import numpy as np
    from stingray.gti import create_gti_from_condition

    lchdulist = pf.open(fits_file)
    lctable = lchdulist[ratehdu].data

    # Units of header keywords
    tunit = lchdulist[ratehdu].header["TIMEUNIT"]

    try:
        mjdref = high_precision_keyword_read(
            lchdulist[ratehdu].header, "MJDREF"
        )
        mjdref = Time(mjdref, scale="tdb", format="mjd")
    except Exception:
        mjdref = None

    try:
        instr = lchdulist[ratehdu].header["INSTRUME"]
    except Exception:
        instr = "EXTERN"

    # ----------------------------------------------------------------
    # Trying to comply with all different formats of fits light curves.
    # It's a madness...
    try:
        tstart = high_precision_keyword_read(
            lchdulist[ratehdu].header, "TSTART"
        )
        tstop = high_precision_keyword_read(lchdulist[ratehdu].header, "TSTOP")
    except Exception:  # pragma: no cover
        raise (Exception("TSTART and TSTOP need to be specified"))

    # For nulccorr lcs this whould work

    timezero = high_precision_keyword_read(
        lchdulist[ratehdu].header, "TIMEZERO"
    )
    # Sometimes timezero is "from tstart", sometimes it's an absolute time.
    # This tries to detect which case is this, and always consider it
    # referred to tstart
    timezero = assign_value_if_none(timezero, 0)

    # for lcurve light curves this should instead work
    if tunit == "d":
        # TODO:
        # Check this. For now, I assume TD (JD - 2440000.5).
        # This is likely wrong
        timezero = Time(2440000.5 + timezero, scale="tdb", format="jd")
        tstart = Time(2440000.5 + tstart, scale="tdb", format="jd")
        tstop = Time(2440000.5 + tstop, scale="tdb", format="jd")
        # if None, use NuSTAR defaulf MJDREF
        mjdref = assign_value_if_none(
            mjdref,
            Time(
                np.longdouble("55197.00076601852"), scale="tdb", format="mjd"
            ),
        )

        timezero = (timezero - mjdref).to("s").value
        tstart = (tstart - mjdref).to("s").value
        tstop = (tstop - mjdref).to("s").value

    if timezero > tstart:
        timezero -= tstart

    time = np.array(lctable.field(timecolumn), dtype=np.longdouble)
    if time[-1] < tstart:
        time += timezero + tstart
    else:
        time += timezero

    try:
        dt = high_precision_keyword_read(lchdulist[ratehdu].header, "TIMEDEL")
        if tunit == "d":
            dt *= 86400
    except Exception:
        warnings.warn(
            "Assuming that TIMEDEL is the median difference between the"
            " light curve times",
            AstropyUserWarning,
        )
        # Avoid NaNs
        good = time == time
        dt = np.median(np.diff(time[good]))

    # ----------------------------------------------------------------
    if ratecolumn is None:
        for name in ["RATE", "RATE1", "COUNTS"]:
            if name in lctable.names:
                ratecolumn = name
                break
        else:  # pragma: no cover
            raise ValueError(
                "None of the accepted rate columns were found in the file")

    rate = np.array(lctable.field(ratecolumn), dtype=float)

    errorcolumn = "ERROR"
    if ratecolumn == "RATE1":
        errorcolumn = "ERROR1"

    try:
        rate_e = np.array(lctable.field(errorcolumn), dtype=np.longdouble)
    except Exception:
        rate_e = np.zeros_like(rate)

    if "RATE" in ratecolumn:
        rate *= dt
        rate_e *= dt

    try:
        fracexp = np.array(lctable.field("FRACEXP"), dtype=np.longdouble)
    except Exception:
        fracexp = np.ones_like(rate)

    good_intervals = (
        (rate == rate) * (fracexp >= fracexp_limit) * (fracexp <= 1)
    )

    rate[good_intervals] /= fracexp[good_intervals]
    rate_e[good_intervals] /= fracexp[good_intervals]

    rate[~good_intervals] = 0

    try:
        gtitable = lchdulist[gtistring].data
        gti_list = np.array(
            [
                [a, b]
                for a, b in zip(
                    gtitable.field("START"), gtitable.field("STOP")
                )
            ],
            dtype=np.longdouble,
        )
    except Exception:
        gti_list = create_gti_from_condition(time, good_intervals)

    lchdulist.close()

    res = {"time": time,
           "counts": rate,
           "err": rate_e,
           "gti": gti_list,
           "mjdref": mjdref.mjd,
           "dt": dt,
           "instr": instr,
           "header": lchdulist[ratehdu].header.tostring()}
    return res


def load_events_and_gtis(
    fits_file,
    additional_columns=None,
    gtistring=None,
    gti_file=None,
    hduname=None,
    column=None,
):
    """Load event lists and GTIs from one or more files.

    Loads event list from HDU EVENTS of file fits_file, with Good Time
    intervals. Optionally, returns additional columns of data from the same
    HDU of the events.

    Parameters
    ----------
    fits_file : str

    Other parameters
    ----------------
    additional_columns: list of str, optional
        A list of keys corresponding to the additional columns to extract from
        the event HDU (ex.: ['PI', 'X'])
    gtistring : str
        Comma-separated list of accepted GTI extensions (default GTI,STDGTI),
        with or without appended integer number denoting the detector
    gti_file : str, default None
        External GTI file
    hduname : str or int, default 1
        Name of the HDU containing the event list
    column : str, default None
        The column containing the time values. If None, we use the name
        specified in the mission database, and if there is nothing there,
        "TIME"
    return_limits: bool, optional
        Return the TSTART and TSTOP keyword values

    Returns
    -------
    retvals : Object with the following attributes:
        ev_list : array-like
            Event times in Mission Epoch Time
        gti_list: [[gti0_0, gti0_1], [gti1_0, gti1_1], ...]
            GTIs in Mission Epoch Time
        additional_data: dict
            A dictionary, where each key is the one specified in additional_colums.
            The data are an array with the values of the specified column in the
            fits file.
        t_start : float
            Start time in Mission Epoch Time
        t_stop : float
            Stop time in Mission Epoch Time
        pi_list : array-like
            Raw Instrument energy channels
        cal_pi_list : array-like
            Calibrated PI channels (those that can be easily converted to energy
            values, regardless of the instrument setup.)
        energy_list : array-like
            Energy of each photon in keV (only for NuSTAR, NICER, XMM)
        instr : str
            Name of the instrument (e.g. EPIC-pn or FPMA)
        mission : str
            Name of the instrument (e.g. XMM or NuSTAR)
        mjdref : float
            MJD reference time for the mission
        header : str
            Full header of the FITS file, for debugging purposes
        detector_id : array-like, int
            Detector id for each photon (e.g. each of the CCDs composing XMM's or
            Chandra's instruments)
    """
    from astropy.io import fits as pf

    hdulist = pf.open(fits_file)
    probe_header = hdulist[0].header
    # Let's look for TELESCOP here. This is the most common keyword to be
    # found in well-behaved headers. If it is not in header 0, I take this key
    # and the remaining information from header 1.
    if "TELESCOP" not in probe_header:
        probe_header = hdulist[1].header
    mission_key = "MISSION"
    if mission_key not in probe_header:
        mission_key = "TELESCOP"
    mission = probe_header[mission_key].lower()

    db = read_mission_info(mission)
    instkey = get_key_from_mission_info(db, "instkey", "INSTRUME")
    instr = mode = None
    if instkey in probe_header:
        instr = probe_header[instkey].strip()

    modekey = get_key_from_mission_info(db, "dmodekey", None, instr)
    if modekey is not None and modekey in probe_header:
        mode = probe_header[modekey].strip()

    gtistring = get_key_from_mission_info(db, "gti", "GTI,STDGTI", instr, mode)
    if hduname is None:
        hduname = get_key_from_mission_info(db, "events", "EVENTS", instr, mode)

    if hduname not in hdulist:
        warnings.warn(f'HDU {hduname} not found. Trying first extension')
        hduname = 1

    datatable = hdulist[hduname].data
    header = hdulist[hduname].header

    ephem = timeref = timesys = None

    if "PLEPHEM" in header:
        ephem = header["PLEPHEM"].strip().lstrip('JPL-').lower()
    if "TIMEREF" in header:
        timeref = header["TIMEREF"].strip().lower()
    if "TIMESYS" in header:
        timesys = header["TIMESYS"].strip().lower()

    if column is None:
        column = get_key_from_mission_info(db, "time", "TIME", instr, mode)
    ev_list = np.array(datatable.field(column), dtype=np.longdouble)

    detector_id = None
    ckey = get_key_from_mission_info(db, "ccol", "NONE", instr, mode)
    if ckey != "NONE":
        detector_id = datatable.field(ckey)
    det_number = None if detector_id is None else list(set(detector_id))

    timezero = np.longdouble(0.)
    if "TIMEZERO" in header:
        timezero = np.longdouble(header["TIMEZERO"])

    ev_list += timezero

    t_start = ev_list[0]
    t_stop = ev_list[-1]
    if "TSTART" in header:
        t_start = np.longdouble(header["TSTART"])
    if "TSTOP" in header:
        t_stop = np.longdouble(header["TSTOP"])

    mjdref = np.longdouble(high_precision_keyword_read(header, "MJDREF"))

    # Read and handle GTI extension
    accepted_gtistrings = gtistring.split(",")

    if gti_file is None:
        # Select first GTI with accepted name
        try:
            gti_list = get_gti_from_all_extensions(
                hdulist,
                accepted_gtistrings=accepted_gtistrings,
                det_numbers=det_number,
            )
        except Exception:  # pragma: no cover
            warnings.warn(
                "No extensions found with a valid name. "
                "Please check the `accepted_gtistrings` values.",
                AstropyUserWarning,
            )
            gti_list = np.array([[t_start, t_stop]], dtype=np.longdouble)
    else:
        gti_list = load_gtis(gti_file, gtistring)

    pi_col = get_key_from_mission_info(db, "ecol", "PI", instr, mode)
    if additional_columns is None:
        additional_columns = [pi_col]
    if pi_col not in additional_columns:
        additional_columns.append(pi_col)

    additional_data = _get_additional_data(datatable, additional_columns)
    hdulist.close()
    # Sort event list
    order = np.argsort(ev_list)
    ev_list = ev_list[order]
    if detector_id is not None:
        detector_id = detector_id[order]

    additional_data = order_list_of_arrays(additional_data, order)

    pi = additional_data[pi_col].astype(np.float32)
    cal_pi = pi

    # EventReadOutput() is an empty class. We will assign a number of attributes to
    # it, like the arrival times of photons, the energies, and some information
    # from the header.
    returns = EventReadOutput()

    returns.ev_list = ev_list
    returns.gti_list = gti_list
    returns.pi_list = pi
    returns.cal_pi_list = cal_pi
    if "energy" in additional_data:
        returns.energy_list = additional_data["energy"]
    else:
        try:
            returns.energy_list = rough_calibration(cal_pi, mission)
        except ValueError:
            returns.energy_list = None
    returns.instr = instr.lower()
    returns.mission = mission.lower()
    returns.mjdref = mjdref
    returns.header = header.tostring()
    returns.additional_data = additional_data
    returns.t_start = t_start
    returns.t_stop = t_stop
    returns.detector_id = detector_id
    returns.ephem = ephem
    returns.timeref = timeref
    returns.timesys = timesys

    return returns


class EventReadOutput():
    def __init__(self):
        pass


def mkdir_p(path):  # pragma: no cover
    """Safe ``mkdir`` function, found at [so-mkdir]_.

    Parameters
    ----------
    path : str
        The absolute path to the directory to be created

    Notes
    -----
    .. [so-mkdir] http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
    """
    import os
    import errno
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def read_header_key(fits_file, key, hdu=1):
    """Read the header key key from HDU hdu of the file ``fits_file``.

    Parameters
    ----------
    fits_file: str
        The file name and absolute path to the event file.

    key: str
        The keyword to be read

    Other Parameters
    ----------------
    hdu : int
        Index of the HDU extension from which the header key to be read.

    Returns
    -------
    value : object
        The value stored under ``key`` in ``fits_file``
    """

    hdulist = fits.open(fits_file, ignore_missing_end=True)
    try:
        value = hdulist[hdu].header[key]
    except KeyError:  # pragma: no cover
        value = ''
    hdulist.close()
    return value


def ref_mjd(fits_file, hdu=1):
    """Read ``MJDREFF``, ``MJDREFI`` or, if failed, ``MJDREF``, from the FITS header.

    Parameters
    ----------
    fits_file : str
        The file name and absolute path to the event file.

    Other Parameters
    ----------------
    hdu : int
        Index of the HDU extension from which the header key to be read.

    Returns
    -------
    mjdref : numpy.longdouble
        the reference MJD
    """

    if isinstance(fits_file, Iterable) and\
            not is_string(fits_file):  # pragma: no cover
        fits_file = fits_file[0]
        logging.info("opening %s" % fits_file)

    hdulist = fits.open(fits_file, ignore_missing_end=True)

    ref_mjd_val = high_precision_keyword_read(hdulist[hdu].header, "MJDREF")

    hdulist.close()
    return ref_mjd_val


def common_name(str1, str2, default='common'):
    """Strip two strings of the letters not in common.

    Filenames must be of same length and only differ by a few letters.

    Parameters
    ----------
    str1 : str
    str2 : str

    Other Parameters
    ----------------
    default : str
        The string to return if ``common_str`` is empty

    Returns
    -------
    common_str : str
        A string containing the parts of the two names in common

    """
    if not len(str1) == len(str2):
        return default
    common_str = ''
    # Extract the MP root of the name (in case they're event files)

    for i, letter in enumerate(str1):
        if str2[i] == letter:
            common_str += letter
    # Remove leading and trailing underscores and dashes
    common_str = common_str.rstrip('_').rstrip('-')
    common_str = common_str.lstrip('_').lstrip('-')
    if common_str == '':
        common_str = default
    logging.debug('common_name: %s %s -> %s' % (str1, str2, common_str))
    return common_str


def split_numbers(number, shift=0):
    """
    Split high precision number(s) into doubles.

    You can specify the number of shifts to move the decimal point.

    Parameters
    ----------
    number: long double
        The input high precision number which is to be split

    Other parameters
    ----------------
    shift: integer
        Move the cut by `shift` decimal points to the right (left if negative)

    Returns
    -------
    number_I: double
        First part of high precision number

    number_F: double
        Second part of high precision number

    Examples
    --------
    >>> n = 12.34
    >>> i, f = split_numbers(n)
    >>> i == 12
    True
    >>> np.isclose(f, 0.34)
    True
    >>> split_numbers(n, 2)
    (12.34, 0.0)
    >>> split_numbers(n, -1)
    (10.0, 2.34)
    """
    if isinstance(number, Iterable):
        number = np.asarray(number)
        number *= 10**shift
        mods = [math.modf(n) for n in number]
        number_F = [f for f, _ in mods]
        number_I = [i for _, i in mods]
    else:
        number *= 10**shift
        number_F, number_I = math.modf(number)

    return np.double(number_I) / 10**shift, np.double(number_F) / 10**shift


def _save_pickle_object(object, filename):
    """
    Save a class object in pickle format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes are saved in a
        dictionary format

    filename: str
        Name of the file in which object is saved
    """

    with open(filename, "wb") as f:
        pickle.dump(object, f)


def _retrieve_pickle_object(filename):
    """
    Retrieves a pickled class object.

    Parameters
    ----------
    filename: str
        Name of the file in which object is saved

    Returns
    -------
    data: class object
    """

    with open(filename, "rb") as f:
        return pickle.load(f)


def _save_hdf5_object(object, filename):
    """
    Save a class object in hdf5 format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes are saved in a
        dictionary format

    filename: str
        Name of the file in which object is saved
    """

    items = vars(object)
    attrs = [name for name in items if items[name] is not None]

    with h5py.File(filename, 'w') as hf:
        for attr in attrs:
            data = items[attr]

            # If data is a single number, store as an attribute.
            if _isattribute(data):
                if isinstance(data, np.longdouble):
                    data_I, data_F = split_numbers(data)
                    names = [attr + '_I', attr + '_F']
                    hf.attrs[names[0]] = data_I
                    hf.attrs[names[1]] = data_F
                else:
                    hf.attrs[attr] = data

            # If data is an array or list, create a dataset.
            else:
                try:
                    if isinstance(data[0], np.longdouble):
                        data_I, data_F = split_numbers(data)
                        names = [attr + '_I', attr + '_F']
                        hf.create_dataset(names[0], data=data_I)
                        hf.create_dataset(names[1], data=data_F)
                    else:
                        hf.create_dataset(attr, data=data)
                except IndexError:
                    # To account for numpy arrays of type 'None' (0-d)
                    pass


def _retrieve_hdf5_object(filename):
    """
    Retrieves an hdf5 format class object.

    Parameters
    ----------
    filename: str
        The name of file with which object was saved

    Returns
    -------
    data: dictionary
        Loads the data from an hdf5 object file and returns
        in dictionary format.
    """

    with h5py.File(filename, 'r') as hf:
        dset_keys = hf.keys()
        attr_keys = hf.attrs.keys()
        data = {}

        dset_copy = list(dset_keys)[:]
        for key in dset_keys:

            # Make sure key hasn't been removed
            if key in dset_copy:
                # Longdouble case
                if key[-2:] in ['_I', '_F']:
                    m_key = key[:-2]
                    # Add integer and float parts
                    data[m_key] = np.longdouble(hf[m_key + '_I'][()])
                    data[m_key] += np.longdouble(hf[m_key + '_F'][()])
                    # Remove integer and float parts from attributes
                    dset_copy.remove(m_key + '_I')
                    dset_copy.remove(m_key + '_F')
                else:
                    data[key] = hf[key][()]

        attr_copy = list(attr_keys)[:]
        for key in attr_keys:

            # Make sure key hasn't been removed
            if key in attr_copy:
                # Longdouble case
                if key[-2:] in ['_I', '_F']:
                    m_key = key[:-2]
                    # Add integer and float parts
                    data[m_key] = np.longdouble(hf.attrs[m_key + '_I'])
                    data[m_key] += np.longdouble(hf.attrs[m_key + '_F'])
                    # Remove integer and float parts from attributes
                    attr_copy.remove(m_key + '_I')
                    attr_copy.remove(m_key + '_F')
                else:
                    data[key] = hf.attrs[key]

    return data


def _save_ascii_object(object, filename, fmt="%.18e", **kwargs):
    """
    Save an array to a text file.

    Parameters
    ----------
    object : numpy.ndarray
        An array with the data to be saved

    filename : str
        The file name to save to

    fmt : str or sequence of strs, optional
        Use for formatting of columns. See `numpy.savetxt` documentation
        for details.

    Other Parameters
    ----------------
    kwargs : any keyword argument taken by `numpy.savetxt`

    """

    try:
        np.savetxt(filename, object, fmt=fmt, **kwargs)
    except TypeError:
        raise Exception("Formatting of columns not recognized! Use 'fmt' "
                        "to format columns including strings or mixed types!")

    pass


def _retrieve_ascii_object(filename, **kwargs):
    """
    Helper function to retrieve ascii objects from file.
    Uses astropy.Table for reading and storing the data.

    Parameters
    ----------
    filename : str
        The name of the file with the data to be retrieved.

    Other Parameters
    -----------------------------
    usecols : {int | iterable}
        The indices of the columns in the file to be returned.
        By default, all columns will be returned

    skiprows : int
        The number of rows at the beginning to skip
        By default, no rows will be skipped.

    names : iterable
        A list of column names to be attached to the columns.
        By default, no column names are added, unless they are specified
        in the file header and can be read by astropy.Table.read
        automatically.

    Returns
    -------
    data : astropy.Table object
        An astropy.Table object with the data from the file
    """
    if not isinstance(filename, str):
        raise TypeError("filename must be string!")

    if 'usecols' in list(kwargs.keys()):
        if np.size(kwargs['usecols']) != 2:
            raise ValueError("Need to define two columns")
        usecols = kwargs["usecols"]
    else:
        usecols = None

    if 'skiprows' in list(kwargs.keys()):
        assert isinstance(kwargs["skiprows"], int)
        skiprows = kwargs["skiprows"]
    else:
        skiprows = 0

    if "names" in list(kwargs.keys()):
        names = kwargs["names"]
    else:
        names = None

    data = Table.read(filename, data_start=skiprows,
                      names=names, format="ascii")

    if usecols is None:
        return data
    else:
        colnames = np.array(data.colnames)
        cols = colnames[usecols]

        return data[cols]


def _save_fits_object(object, filename, **kwargs):
    """
    Save a class object in fits format.

    Parameters
    ----------
    object: class instance
        A class object whose attributes would be saved in a dictionary format.

    filename: str
        The file name to save to

    Additional Keyword Parameters
    -----------------------------
    tnames: str iterable
        The names of HDU tables. For instance, in case of eventlist,
        tnames could be ['EVENTS', 'GTI']

    colsassign: dictionary iterable
        This indicates the correct tables to which to assign columns
        to. If this is None or if a column is not provided, it/they will
        be assigned to the first table.

        For example, [{'gti':'GTI'}] indicates that gti values should be
        stored in GTI table.
    """

    tables = []

    if 'colsassign' in list(kwargs.keys()):
        colsassign = kwargs['colsassign']
        iscolsassigned = True
    else:
        iscolsassigned = False

    if 'tnames' in list(kwargs.keys()):
        tables = kwargs['tnames']
    else:
        tables = ['MAIN']

    items = vars(object)
    attrs = [name for name in items if items[name] is not None]

    cols = []
    hdrs = []

    for t in tables:
        cols.append([])
        hdrs.append(fits.Header())

    for attr in attrs:
        data = items[attr]

        # Get the index of table to which column belongs
        if iscolsassigned and attr in colsassign.keys():
            index = tables.index(colsassign[attr])
        else:
            index = 0

        # If data is a single number, store as metadata
        if _isattribute(data):
            if isinstance(data, np.longdouble):
                # Longdouble case. Split and save integer and float parts
                data_I, data_F = split_numbers(data)
                names = [attr + '_I', attr + '_F']
                hdrs[index][names[0]] = data_I
                hdrs[index][names[1]] = data_F
            else:
                # Normal case. Save as it is
                hdrs[index][attr] = data

        # If data is an array or list, insert as table column
        else:
            try:
                if isinstance(data[0], np.longdouble):
                    # Longdouble case. Split and save integer and float parts
                    data_I, data_F = split_numbers(data)
                    names = [attr + '_I', attr + '_F']
                    cols[index].append(
                        fits.Column(name=names[0],
                                    format='D',
                                    array=data_I))
                    cols[index].append(
                        fits.Column(name=names[1],
                                    format='D',
                                    array=data_F))
                else:
                    # Normal case. Save as it is
                    cols[index].append(
                        fits.Column(name=attr,
                                    format=_lookup_format(data[0]),
                                    array=data))
            except IndexError:
                # To account for numpy arrays of type 'None' (0-d)
                pass

    tbhdu = fits.HDUList()

    # Create binary tables
    for i in range(0, len(tables)):
        if len(cols[i]) > 0:
            tbhdu.append(fits.BinTableHDU.from_columns(cols[i],
                                                       header=hdrs[i],
                                                       name=tables[i]))

    tbhdu.writeto(filename)


def _retrieve_fits_object(filename, **kwargs):
    """
    Retrieves a fits format class object.

    Parameters
    ----------
    filename: str
        The name of file with which object was saved

    Other Parameters
    ----------------
    cols: str iterable
        The names of columns to extract from fits tables.

    Returns
    -------
    data: dictionary
        Loads the data from a fits object file and returns
        in dictionary format.
    """

    data = {}

    if 'cols' in list(kwargs.keys()):
        cols = [col.upper() for col in kwargs['cols']]
    else:
        cols = []

    with fits.open(filename, memmap=False, ignore_missing_end=True) as hdulist:
        fits_cols = []

        # Get columns from all tables
        for i in range(1, len(hdulist)):
            fits_cols.append([h.upper() for h in hdulist[i].data.names])

        for c in cols:
            for i in range(0, len(fits_cols)):
                # .upper() is used because `fits` stores values in upper case
                hdr_keys = [h.upper() for h in hdulist[i + 1].header.keys()]

                # Longdouble case. Check for columns
                if c + '_I' in fits_cols[i] or c + '_F' in fits_cols[i]:
                    if c not in data.keys():
                        data[c] = np.longdouble(hdulist[i + 1].data[c + '_I'])
                        data[c] += np.longdouble(hdulist[i + 1].data[c + '_F'])

                # Longdouble case. Check for header keys
                if c + '_I' in hdr_keys or c + '_F' in hdr_keys:
                    if c not in data.keys():
                        data[c] = \
                            np.longdouble(hdulist[i + 1].header[c + '_I'])
                        data[c] += \
                            np.longdouble(hdulist[i + 1].header[c + '_F'])

                # Normal case. Check for columns
                elif c in fits_cols[i]:
                    data[c] = hdulist[i + 1].data[c]

                # Normal case. Check for header keys
                elif c in hdr_keys:
                    data[c] = hdulist[i + 1].header[c]
        hdulist.close()
    return data


def _lookup_format(var):
    """
    Looks up relevant format in fits.

    Parameters
    ----------
    var : object
        An object to look up in the table

    Returns
    -------
    lookup : str
        The str describing the type of ``var``
    """

    lookup = {"<type 'int'>": "J", "<type 'float'>": "E",
              "<type 'numpy.int64'>": "K", "<type 'numpy.float64'>": "D",
              "<type 'numpy.float128'>": "D", "<type 'str'>": "30A",
              "<type 'bool'": "L"}

    form = type(var)

    try:
        return lookup[str(form)]
    except KeyError:
        # If an entry is not contained in lookup dictionary
        return "D"


def _isattribute(data):
    """
    Check if data is a single number or an array.

    Parameters
    ----------
    data : object
        The object to be checked.

    Returns:
        bool
        True if the data is a single number, False if it is an iterable.
    """

    if isinstance(data, Iterable) and not isinstance(data, (str, bytes)):
        return False
    else:
        return True


def write(input_, filename, format_='pickle', **kwargs):
    """
    Pickle a class instance. For parameters depending on
    ``format_``, see individual function definitions.

    Parameters
    ----------
    object: a class instance
        The object to be stored

    filename: str
        The name of the file to be created

    format_: str
        The format in which to store file. Formats supported
        are ``pickle``, ``hdf5``, ``ascii`` or ``fits``
    """

    if format_ == 'pickle':
        _save_pickle_object(input_, filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            _save_hdf5_object(input_, filename)
        else:
            utils.simon('h5py not installed, using pickle instead'
                        'to save object.')
            _save_pickle_object(input_, filename.split('.')[0] +
                                '.pickle')

    elif format_ == 'ascii':
        _save_ascii_object(input_, filename, **kwargs)

    elif format_ == 'fits':
        _save_fits_object(input_, filename, **kwargs)

    else:
        utils.simon('Format not understood.')


def read(filename, format_='pickle', **kwargs):
    """
    Return a saved class instance.

    Parameters
    ----------
    filename: str
        The name of the file to be retrieved.

    format_: str
        The format used to store file. Supported formats are
        pickle, hdf5, ascii or fits.

    Returns
    -------
    data : {``object`` | ``astropy.table`` | ``dict``}

        * If ``format_`` is ``pickle``, an object is returned.
        * If ``format_`` is ``ascii``, `astropy.table` object is returned.
        * If ``format_`` is ``hdf5`` or 'fits``, a dictionary object is returned.
    """

    if format_ == 'pickle':
        return _retrieve_pickle_object(filename)

    elif format_ == 'hdf5':
        if _H5PY_INSTALLED:
            return _retrieve_hdf5_object(filename)
        else:
            utils.simon('h5py not installed, cannot read an'
                        'hdf5 object.')

    elif format_ == 'ascii':
        return _retrieve_ascii_object(filename, **kwargs)

    elif format_ == 'fits':
        return _retrieve_fits_object(filename, **kwargs)

    else:
        utils.simon('Format not understood.')


def savefig(filename, **kwargs):
    """
    Save a figure plotted by ``matplotlib``.

    Note : This function is supposed to be used after the ``plot``
    function. Otherwise it will save a blank image with no plot.

    Parameters
    ----------
    filename : str
        The name of the image file. Extension must be specified in the
        file name. For example filename with `.png` extension will give a
        rasterized image while ``.pdf`` extension will give a vectorized
        output.

    kwargs : keyword arguments
        Keyword arguments to be passed to ``savefig`` function of
        ``matplotlib.pyplot``. For example use `bbox_inches='tight'` to
        remove the undesirable whitepace around the image.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required for savefig()")

    if not plt.fignum_exists(1):
        utils.simon("use ``plot`` function to plot the image first and "
                    "then use ``savefig`` to save the figure.")

    plt.savefig(filename, **kwargs)

# bayespecfit/flux_model.py

import numpy as np
import pytensor.tensor as pt

from numpy.typing import ArrayLike, NDArray
from pytensor.tensor.variable import TensorVariable

###################### conversion between wav and vel ##########################

SPEED_OF_LIGHT = 2.99792458e5


def velocity_rf(lambda_rf: ArrayLike, lambda_0: float) -> ArrayLike:
    """convert rest-frame wavelength to relative velocity"""
    lambda_rf = np.asarray(lambda_rf)
    v = SPEED_OF_LIGHT * ((lambda_rf / lambda_0) ** 2 - 1) / ((lambda_rf / lambda_0) ** 2 + 1)

    return v


def wv_rf(vel_rf: ArrayLike, lambda_0: float) -> ArrayLike:
    """convert relative velocity to rest-frame wavelength"""
    vel_rf = np.asarray(vel_rf)
    beta = vel_rf / SPEED_OF_LIGHT
    wv = lambda_0 * ((1 + beta) / (1 - beta)) ** 0.5

    return wv


def velocity_rf_line(lambda_0: float, lambda_1: float, vel: float) -> float:
    """get the relative velocity of a feature assuming another line

    Parameters
    ----------
    lambda_0 : float
        wavelength of the original line [angstrom]
    lambda_1 : float
        wavelength of the new line [angstrom]
    vel : float
        the velocity relative to lambda_0 [km/s]

    Returns
    -------
    vel_1 : float
        the velocity relative to lambda_1 [km/s]
    """

    lambda_rf = ((vel / SPEED_OF_LIGHT + 1) / (-vel / SPEED_OF_LIGHT + 1)) ** 0.5 * lambda_0
    vel_1 = velocity_rf(lambda_rf, lambda_1)

    return vel_1


###################### spectroscopic models ##########################

def contains_tensor_variable(*args):
    """Check if any argument is or contains a TensorVariable."""
    for arg in args:
        if isinstance(arg, (pt.TensorVariable, pt.TensorConstant)):
            return True
        
        if isinstance(arg, (list, tuple)):
            for item in arg:
                if contains_tensor_variable(item):
                    return True
    
    return False

def _calc_gauss(vel_mean, sig_vel, amplitude, vel_rf):
    """Gaussian profile"""
    gauss = amplitude / np.sqrt(2 * np.pi * sig_vel**2) * np.exp(-0.5 * (vel_rf - vel_mean) ** 2 / sig_vel**2)
    return gauss


def _calc_lorentzian(vel_mean, gamma_vel, amplitude, vel_rf):
    """Lorentzian profile"""
    lorentz = amplitude / np.sqrt(np.pi * gamma_vel) / (1 + (vel_rf - vel_mean) ** 2 / gamma_vel**2)
    return lorentz


def calc_model_flux(
    wv_rf: ArrayLike | TensorVariable,
    fl_blue: ArrayLike | TensorVariable,
    fl_red: ArrayLike | TensorVariable,
    vel_mean: ArrayLike | TensorVariable,
    log_vel_sig: ArrayLike | TensorVariable,
    log_amp: ArrayLike | TensorVariable,
    lines: list,
    rel_strength: list[float | TensorVariable],
    line_regions: list[tuple],
    vel_resolution: list[float],
    model: str = "Gauss",
) -> NDArray | TensorVariable:
    """Calculate normalized flux based on a Gaussian/Lorentzian model.
    
    This version handles both NumPy and PyTensor inputs:
    - If inputs contain TensorVariables, returns a TensorVariable
    - If inputs contain only regular arrays/floats, returns a NumPy array
    """
    # Detect if any input is a TensorVariable
    using_tensor = contains_tensor_variable(
        wv_rf, fl_blue, fl_red, vel_mean, log_vel_sig, log_amp, rel_strength
    )

    # Choose the correct implementation based on the model
    if model == "Gauss":
        calc = _calc_gauss
    elif model == "Lorentz":
        calc = _calc_lorentzian
    else:
        raise NameError("Line model not supported (optional: Gauss & Lorentz)")
    
    # Choose the implementation based on input types
    if using_tensor:
        return _calc_model_flux_tensor(
            wv_rf, fl_blue, fl_red, vel_mean, log_vel_sig, log_amp,
            lines, rel_strength, line_regions, vel_resolution, calc
        )
    else:
        return _calc_model_flux_numpy(
            wv_rf, fl_blue, fl_red, vel_mean, log_vel_sig, log_amp,
            lines, rel_strength, line_regions, vel_resolution, calc
        )


def _calc_model_flux_numpy(
    wv_rf, fl_blue, fl_red, vel_mean, log_vel_sig, log_amp,
    lines, rel_strength, line_regions, vel_resolution, calc
):
    """NumPy implementation for non-tensor inputs."""
    # Convert inputs to numpy arrays
    wv_rf = np.asarray(wv_rf)
    fl_blue = np.asarray(fl_blue)
    fl_red = np.asarray(fl_red)
    vel_mean = np.asarray(vel_mean)
    log_vel_sig = np.asarray(log_vel_sig)
    log_amp = np.asarray(log_amp)
    
    # Create a 2D mask: regions x wv_values
    wv_blue = np.array([line[0] for line in line_regions])
    wv_red = np.array([line[1] for line in line_regions])
    in_line_region = (wv_rf >= wv_blue[:, np.newaxis]) & (wv_rf <= wv_red[:, np.newaxis])
    region_idx = np.argmax(in_line_region, axis=0)

    ##################### get the continuum flux (a piecewise linear function) ############################

    # Calculate slopes for all segments at once
    slopes = np.divide(fl_red - fl_blue, wv_red - wv_blue, out=np.zeros_like(fl_red), where=(wv_red != wv_blue))

    # Calculate interpolated values (assumes non-overlapping regions)
    continuum = fl_blue[region_idx] + slopes[region_idx] * (wv_rf - wv_blue[region_idx])

    ###################################### get the absorption lines ########################################

    # The spectral resolution of the spectrograph limits the line width
    vel_res = np.array(vel_resolution)[region_idx]

    absorption = np.zeros_like(wv_rf)

    sig_vel = 10 ** log_vel_sig
    amp = 10 ** log_amp

    for k in range(len(lines)):
        sig_instru = (sig_vel[k]**2 + vel_res**2) ** 0.5

        for rel_s, li in zip(rel_strength[k], lines[k][:-1]):
            vel_rf = velocity_rf(wv_rf, li)
            absorption += rel_s * calc(vel_mean[k], sig_instru, amp[k], vel_rf)

        # The last line: rel_s is fixed to 1
        vel_rf = velocity_rf(wv_rf, lines[k][-1])
        absorption += calc(vel_mean[k], sig_instru, amp[k], vel_rf)

    model_flux = continuum * (1 - absorption)

    return model_flux


def _calc_model_flux_tensor(
    wv_rf, fl_blue, fl_red, vel_mean, log_vel_sig, log_amp,
    lines, rel_strength, line_regions, vel_resolution, calc
):
    """PyTensor implementation for tensor inputs."""
    # Convert inputs to tensor variables
    wv_rf = pt.as_tensor_variable(wv_rf)
    fl_blue = pt.as_tensor_variable(fl_blue)
    fl_red = pt.as_tensor_variable(fl_red)
    vel_mean = pt.as_tensor_variable(vel_mean)
    log_vel_sig = pt.as_tensor_variable(log_vel_sig)
    log_amp = pt.as_tensor_variable(log_amp)
    
    # Extract line region boundaries
    wv_blue = pt.as_tensor_variable([line[0] for line in line_regions])
    wv_red = pt.as_tensor_variable([line[1] for line in line_regions])
    
    ##################### get the continuum flux (a piecewise linear function) ############################
    
    # Initialize continuum with zeros
    continuum = pt.zeros_like(wv_rf)
    
    # Calculate slopes safely avoiding division by zero
    slopes = pt.switch(
        pt.eq(wv_red - wv_blue, 0),
        pt.zeros_like(fl_red),
        (fl_red - fl_blue) / (wv_red - wv_blue)
    )
    
    # Apply piecewise linear function for each region
    for i in range(len(line_regions)):
        # Create mask for this region
        region_mask = pt.and_(
            pt.ge(wv_rf, wv_blue[i]),
            pt.le(wv_rf, wv_red[i])
        )
        
        # Calculate values for this region
        region_values = fl_blue[i] + slopes[i] * (wv_rf - wv_blue[i])
        
        # Update continuum where mask is True
        continuum = pt.switch(region_mask, region_values, continuum)
    
    ###################################### get the absorption lines ########################################
    
    # Initialize vel_res tensor for each wavelength point
    vel_res_tensor = pt.zeros_like(wv_rf)
    
    # Set velocity resolution for each region
    for i in range(len(line_regions)):
        region_mask = pt.and_(
            pt.ge(wv_rf, wv_blue[i]),
            pt.le(wv_rf, wv_red[i])
        )
        vel_res_tensor = pt.switch(region_mask, vel_resolution[i], vel_res_tensor)
    
    # Initialize absorption
    absorption = pt.zeros_like(wv_rf)
    
    # Convert log values to linear values
    sig_vel = 10 ** log_vel_sig
    amp = 10 ** log_amp
    
    # Loop through lines
    for k in range(len(lines)):
        # Calculate instrumental broadening
        sig_instru = pt.sqrt(sig_vel[k]**2 + vel_res_tensor**2)
        
        # Loop through components
        for j, (rel_s, li) in enumerate(zip(rel_strength[k], lines[k][:-1])):
            # Convert wavelength to velocity
            vel_rf = velocity_rf(wv_rf, li)
            rel_s_tensor = pt.as_tensor_variable(rel_s)
            
            # Calculate absorption profile
            profile = calc(vel_mean[k], sig_instru, amp[k], vel_rf)
                
            # Add to total absorption
            absorption = absorption + rel_s_tensor * profile

        # The last line: rel_s is fixed to 1
        vel_rf = velocity_rf(wv_rf, lines[k][-1])
        absorption = absorption + calc(vel_mean[k], sig_instru, amp[k], vel_rf)
    
    # Calculate final flux
    model_flux = continuum * (1 - absorption)
    
    return model_flux
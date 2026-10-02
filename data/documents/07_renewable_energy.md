---
title: Renewable Energy from Solar and Wind
doc_id: renewable_energy
topic: engineering
source: Adaptive-RAG teaching corpus (original text)
---

## Overview

Renewable energy is energy drawn from sources that are replenished on a human timescale.
Solar and wind now dominate new generating capacity worldwide, largely because their
costs have fallen faster than any forecast made twenty years ago. Both are ultimately
solar in origin: sunlight arrives directly as radiation, and wind exists because uneven
heating of the surface creates pressure differences that set air in motion.

The quantity of energy available is not the constraint. The solar radiation reaching the
surface of the planet in roughly ninety minutes is comparable to total annual human
energy consumption. The engineering problem is that this energy is dilute, intermittent
and unevenly distributed in both space and time, so the difficulty lies in collection,
conversion and matching supply to demand rather than in supply itself.

## Photovoltaic Conversion

A photovoltaic cell converts light directly into electricity using the photovoltaic
effect. Most commercial cells are made from crystalline silicon doped to create a p-n
junction. When a photon with energy above the band gap of the material is absorbed, it
lifts an electron into the conduction band and leaves a positively charged hole behind.
The built-in electric field at the junction sweeps the two in opposite directions, and
the resulting separation of charge drives a current through an external circuit.

Efficiency is bounded by fundamental physics. Photons with energy below the band gap
pass through unabsorbed, while the excess energy of high-energy photons is lost as heat.
Together these losses give the Shockley-Queisser limit of roughly thirty-three per cent
for a single-junction cell, and commercial silicon modules typically achieve twenty to
twenty-three per cent. Output falls as cells heat up, so a bright cool day can yield more
than a bright hot one. Multi-junction cells stack materials with different band gaps to
capture a wider slice of the spectrum and exceed the single-junction limit, but they are
expensive and mostly reserved for spacecraft and concentrator systems.

## Wind Turbines

A wind turbine extracts kinetic energy from moving air. The power available in the wind
is proportional to the swept area of the rotor and to the cube of the wind speed, so
doubling the wind speed increases the available power eightfold. This cubic relationship
explains why site selection dominates the economics of a wind farm and why turbine
towers have grown steadily taller, since wind speed increases with height above the
rough surface layer.

No turbine can capture all of the energy in the wind, because the air must retain enough
speed to leave the rotor and make way for the air behind it. The theoretical maximum
fraction that can be extracted is about fifty-nine per cent, a result known as the Betz
limit, and modern three-bladed horizontal-axis machines reach around three quarters of
it. Turbines are designed to a rated power reached at a moderate wind speed; above that
the blades are pitched to spill excess energy and protect the drivetrain, and beyond a
cut-out speed the machine shuts down entirely. Offshore installations benefit from
stronger and steadier wind and from fewer constraints on size, but pay for it in
foundation and maintenance costs.

## Intermittency, Storage and Grid Integration

Solar and wind are variable rather than unreliable: their output is predictable in
aggregate but not controllable. A useful measure is the capacity factor, the ratio of
actual annual output to what the plant would produce running continuously at rated
power. Typical values are around ten to twenty-five per cent for solar photovoltaics
depending on latitude, twenty-five to forty per cent for onshore wind and forty to sixty
per cent for modern offshore wind. Because output does not follow demand, matching the
two becomes the central system problem as the renewable share grows.

Several strategies address this. Geographical dispersion and interconnection smooth
local fluctuations, since it is rarely calm everywhere at once. Combining technologies
helps, because wind generation in mid-latitudes tends to peak in winter when solar
output is lowest. Storage shifts energy in time: lithium-ion batteries are well suited to
hours of shifting and to rapid frequency response, pumped hydroelectric storage remains
the largest form of grid storage by capacity, and hydrogen produced by electrolysis is
under investigation for seasonal storage despite poor round-trip efficiency. Demand-side
management moves flexible loads such as vehicle charging and water heating towards
periods of surplus. Grids also need to replace the mechanical inertia formerly supplied
by large synchronous generators, which is now increasingly provided by inverters
programmed to respond to changes in frequency within milliseconds.

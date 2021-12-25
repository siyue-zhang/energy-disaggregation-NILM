# energy-disaggregation-NILM

Power sub-metering can bring a lot of benefits to residential customers. Instead of a bulk value of monthly electricity consumption given by the utility bill, sub-metering provides itemized energy consumption data by individual appliances. Consumers are awared of on-off status of appliances and power consumption in the real time. They can identify the abnormal use of appliances timely, monitor the use remotely, avoid the unnecessary use and improve the energy efficiency.

However, sub-metering by installing sensors or meters is not always economic in the appliance level. Data-driven nonintrusive load monitoring (NILM) is a low-cost alternative to smart metering to identify granular energy consumption data at appliance-level from whole-home energy data. NILM technology identifies the unique electrical signature (i.e. series of power changes) of appliances, and separates the individual power curve of the appliance out of the aggregate power curve.

![](http://nilmtk.github.io/img/submetered.png)

## REFIT Dataset

Many public load measurement datasets are available for research. This project primarily focuses on the [REFIT](http://dx.doi.org/10.1038/sdata.2016.122) Electrical Load Measurements dataset which includes cleaned electrical consumption data in Watts for 20 households at aggregate and appliance level, timestamped and sampled at 8 second intervals. This dataset is intended to be used for research into energy conservation and advanced energy services, ranging from non-intrusive appliance load monitoring, demand response measures, tailored energy and retrofit advice, appliance usage analysis, consumption and time-use statistics and smart home/building automation. 

Download: https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

## Open source NILM toolkit

This proejct uses [nilmtk](http://nilmtk.github.io/) python library to preprocess the raw data. This tool is designed specifically to enable the comparison of energy disaggregation algorithms in a reproducible manner. For example, plot_wen_on function in `nilmtk` visualizes when the appliances are above the defined power threshold in House 2 as the figure below. 

![time-use]("./images/time-use.png")

#-----------------------Project Data Helper Functions---------------------------
#-Author: A. Jordan Nafa------------------------------Created: August 11, 2021-#
#-R Version: 4.1.0----------------------------------Revised: December 18, 2021-#

# Custom theme for data visualizations
plot_theme <- function(...) {
  theme_bw() + theme(
    # Set the outer margins of the plot to 1/5 of an inch on all sides
    #plot.margin = margin(0.2, 0.2, 0.2, 0.2, "in"),
    # Specify the default settings for the plot title
    plot.title = element_text(
      size = 22,
      face = "bold",
      family = "serif"
    ),
    # Specify the default settings for caption text
    plot.caption = element_text(
      size = 12,
      family = "serif"
    ),
    # Specify the default settings for subtitle text
    plot.subtitle = element_text(
      size = 16,
      family = "serif"
    ),
    # Specify the default settings for axis titles
    axis.title = element_text(
      size = 18,
      face = "bold",
      family = "serif"
    ),
    # Specify the default settings specific to the x axis title
    axis.title.y = element_text(margin = margin(r = 10, l = -10)),
    # Specify the default settings specific to the y axis title
    axis.title.x = element_text(margin = margin(t = 10, b = -10)),
    # Specify the default settings for x axis text
    axis.text.x = element_text(
      size = 12,
      family = "serif"
    ),
    # Specify the default settings for y axis text
    axis.text.y = element_text(
      size = 12,
      family = "serif"
    ),
    # Specify the default settings for legend titles
    legend.title = element_text(
      size = 16,
      face = "bold",
      family = "serif"
    ),
    # Specify the default settings for legend text
    legend.text = element_text(
      size = 14,
      family = "serif"
    ),
    # Additional Settings Passed to theme()
    ...
  )
}

# Custom theme for spatial visualizations
map_theme <- function(caption.hjust = 1, caption.vjust = 0, ...) {
  theme_void() + theme(
    # Set the outer margins of the plot to 1/5 of an inch on all sides
    plot.margin = margin(0.2, 0.2, 0.2, 0.2, "in"),
    # Specify the default settings for the plot title
    plot.title = element_text(
      size = 22,
      face = "bold",
      family = "serif"
    ),
    # Specify the default settings for caption text
    plot.caption = element_text(
      size = 12,
      family = "serif",
      hjust = caption.hjust,
      vjust = caption.vjust
    ),
    # Specify the default settings for subtitle text
    plot.subtitle = element_text(
      size = 16,
      family = "serif"
    ),
    # Specify the default settings for axis titles
    axis.title = element_blank(),
    # Specify the default settings for axis text
    axis.text = element_blank(),
    # Specify the default settings for legend titles
    legend.title = element_text(
      size = 16,
      face = "bold",
      family = "serif"
    ),
    # Specify the default settings for legend text
    legend.text = element_text(
      size = 14,
      family = "serif"
    ),
    # Font Settings for Face Strips
    strip.text = element_text(
      size = 14,
      face = "bold",
      family = "serif"
    ),
    # Additional Settings Passed to theme()
    ...
  )
}

# A Function  for creating a field containing the meta data from a brms object----
stan_metadata <- function(x, ...){
  # Construct a field with relevant metadata
  x$meta_data <- map_dfr(
    .x = x$fit@stan_args,
    .f = ~ tibble(
      warmup = str_remove_all(.x$time_info[1], "[^[:digit:]|\\.]"),
      sampling = str_remove_all(.x$time_info[2], "[^[:digit:]|\\.]"),
      total = str_remove_all(.x$time_info[3], "[^[:digit:]|\\.]"),
      misc = str_remove_all(.x$time_info[4], "[^[:digit:]|\\.]"),
      metadata = c(
        str_c("stanc_version:", .x$stanc_version[1], sep = " "),
        str_c("opencl_device_name:", .x$opencl_device_name[1], sep = " "),
        str_c("opencl_platform_name", .x$opencl_platform_name[1], sep = " "),
        str_c("date", .x$start_datetime[1], sep = " ")
      )
    ),
    .id = "chain"
  )
}

# A function to calulate median odds ratios for GLMMs in brms-----
median_or <- function(re_var, ...){
  # Calculate the estimate for the median odds ratio
  mor_point <- exp(sqrt(2*(re_var[1]^2))*qnorm(.75))
  # Calculate the lower CI for the median odds ratio
  mor_lower <- exp(sqrt(2*(re_var[2]^2))*qnorm(.75))
  # Calculate the upper CI for the median odds ratio
  mor_upper <- exp(sqrt(2*(re_var[3]^2))*qnorm(.75))
  # Format it all as pretty tibble
  mor_ests <- tribble(
    ~ Median, ~ Lower, ~ Upper,
    mor_point, mor_lower, mor_upper
  )
  # Return the formatted tibble
  return(mor_ests)
}

# Arguments: 
#   rev_var   A numeric vector of length 3 consisting of the variance
#             estimates to use in the calculation. Must be ordered
#             as c(sd_median, sd_lower, sd_upper)
#
#   ...       Currently unused

# A function to calculate interval odds ratios for GLMMs in brms-----
interval_or <- function(re_sd, cluster_coef, values, interval = c(0.1, 0.9), ...){
  # Calculate the difference in values of `cluster_coef` to use
  diff <- values[1] - values[2]
  # Calculate the lower IOR
  lower_ior <- exp(cluster_coef*diff + sqrt(2*(re_sd^2))*qnorm(interval[1]))
  # Calculate the upper IOR
  upper_ior <- exp(cluster_coef*diff + sqrt(2*(re_sd^2))*qnorm(interval[2]))
  # Format it all as pretty tibble
  ior_ests <- tribble(
    ~ `Coefficient`, ~ `Lower IOR`, ~ `Upper IOR`, 
    cluster_coef, lower_ior, upper_ior
  )
  # Return the formatted tibble
  return(ior_ests)
}

# Arguments: 
#   rev_sd       A numeric vector of length 3 consisting of the variance
#                estimates to use in the calculation. Must be ordered
#                as c(sd_median, sd_lower, sd_upper)
#
#  cluster_coef  The coefficient for the group-level predictor to 
#                calculate the IOR for 
#
#  values        A numeric vector of length 2 for which to calculate the
#                difference.
#
# interval       A numeric vector of length 2 containing the interval to
#                be used for the upper and lower bounds. Defaults to c(0.1, 0.9)
#                which corresponds to an 80% IOR
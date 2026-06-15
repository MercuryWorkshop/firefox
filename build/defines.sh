# Define indicating that this build is prior to one of the early betas. To be
# unset mid-way through the beta cycle.
# Unset for this embedding: leaving it on makes MOZ_DIAGNOSTIC_ASSERT live
# throughout SpiderMonkey/Gecko hot paths (GC barriers, object/shape ops, CacheIR
# that PBL runs), which is pure overhead for our interpreter-only build, and also
# turns benign nightly-only assertions into engine-killing crashes on real sites.
# Empty value => is_early_beta_or_earlier stays unset => the define is not set
# (release-channel behavior).
EARLY_BETA_OR_EARLIER=

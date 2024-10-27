from dataclasses import dataclass
import modules.shared


@dataclass()
class Opts:
    hide_samplers = []
    disable_mmap_load_safetensors = False
    sd_checkpoint_cache = 0
    hide_ldm_prints = True
    data = {}
    sd_vae_overrides_per_model_preferences = True
    sd_vae = "Automatic"
    sd_vae_checkpoint_cache = 0
    textual_inversion_print_at_load = False
    s_min_uncond = 0.0
    s_churn = 0.0
    s_tmin = 0.0
    s_tmax = 0.0
    s_noise = 1.0
    use_old_emphasis_implementation = False
    emphasis = "Original"
    CLIP_stop_at_last_layers = 1
    textual_inversion_add_hashes_to_infotext = True
    face_restoration = False
    tiling = False
    disable_all_extensions: str = "none"
    restore_config_state_file = ""
    live_previews_enable = False
    show_progress_type = "Approx NN"
    sd_unet = "Automatic"
    randn_source = "GPU"
    token_merging_ratio = 0.0
    token_merging_ratio_hr = 0.0
    eta_noise_seed_delta = 0
    add_model_hash_to_info = True
    add_model_name_to_info = True
    fp8_storage: str = "Disable"
    cache_fp16_weight: bool = False
    auto_backcompat: bool = True
    add_vae_hash_to_info = True
    add_vae_name_to_info = True
    face_restoration_model = None
    add_version_to_infotext: bool = True
    add_user_name_to_info = True
    use_old_scheduling = False
    sdxl_crop_left = 0
    sdxl_crop_top = 0
    enable_quantization = False
    always_discard_next_to_last_sigma: bool = False
    k_sched_type = "Automatic"
    use_old_karras_scheduler_sigmas = False
    img2img_extra_noise = 0.0
    sgm_noise_multiplier = False
    show_progress_every_n_steps = 1
    multiple_tqdm = True
    disable_console_progressbars = False
    sd_vae_decode_method = "Full"
    samples_save = True
    save_incomplete_images = False
    overlay_inpaint = True
    samples_format = "png"
    grid_save_to_dirs = True
    save_to_dirs = True
    directories_filename_pattern = "[date]"
    samples_filename_pattern = None
    save_images_add_number = True
    enable_pnginfo = True
    jpeg_quality = 80
    save_images_replace_action = "Replace"
    target_side_length = 4000
    export_for_4chan = True
    img_downscale_threshold = 4.0
    save_txt = True
    grid_only_if_multiple = True
    return_grid = True
    grid_save = True
    comma_padding_backtrack = 20
    # img2img
    inpaintng_mask_weight = 1.0
    initial_noise_multiplier = 1.0
    img2img_color_correction = False
    save_init_img = False
    img2img_background_color = "#ffffff"
    upscaler_for_img2img = None
    sd_vae_encode_method = "Full"
    token_merging_ratio_img2img = 0.0
    token_merging_ratio_hr = 0.0
    img2img_fix_steps = False
    sdxl_refiner_high_aesthetic_score = 6.0
    sdxl_refiner_low_aesthetic_score = 2.5
    no_dpmpp_sde_batch_determinism = False
    forge_try_reproduce = "None"
    sigma_min = 0
    sigma_max = 0
    rho = 0
    skip_early_cond = 0
    s_min_uncond_all = False


modules.shared.opts = Opts()
modules.shared.options_templates = {}

from simplediffusion import main


main()

PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction

Paper

[PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction Isaac Deutsch*, Nicolas Moenne-Loccoz ¨ *, Gavriel State, Zan Gojcic NVIDIA {ideutsch, nicolasm, gstate, zgojcic}@nvidia.com https://research.nvidia.com/labs/sil/projects/ppisp/ Figure 1. We introduce a differentiable image processing pipeline applied to radiance field reconstruction. By modeling the behavior of conventional cameras, our approach disentangles image formation effects from the rest of the pipeline. Our physically-plausible model admits a controller module that predicts exposure and color changes for novel views. Abstract Multi-view 3D reconstruction methods remain highly sensitive to photometric inconsistencies arising from camera optical characteristics and variations in image signal processing (ISP). Existing mitigation strategies such as per-frame latent variables or affine color corrections lack physical grounding and generalize poorly to novel views. We propose the Physically-Plausible ISP (PPISP) correction module, which disentangles camera-intrinsic and capture-dependent effects through physically based and interpretable transformations. A dedicated PPISP controller, trained on the input views, predicts ISP parameters for novel viewpoints, analogous to auto exposure and auto white balance in real cameras. This design enables realistic and fair evaluation on novel views without access to ground-truth images. PPISP achieves SoTA performance on standard benchmarks, while providing intuitive control and supporting the integration of metadata when available. The source code is available at: https://github.com/ nv-tlabs/ppisp ∗ Equal contribution. 1. Introduction State-of-the-art multi-view 3D reconstruction methods have significantly advanced the fidelity of novel view synthesis (NVS), transforming it into a technology with real-world applications in physical AI simulation, virtual production, and content creation. Despite these advances, the quality of reconstruction and view synthesis remains highly sensitive to the quality of the input data-both to the distribution of camera poses and to multi-view appearance inconsistencies. The latter often arise from variations in camera optical characteristics and image signal processing (ISP) settings over time. These variations result in differences in color tone, intensity, and contrast that violate the photometric consistency assumptions underlying 3D reconstruction. A common strategy to mitigate these appearance variations is to introduce additional, optimizable per-frame or per-camera parameters designed to capture photometric residuals while preserving a consistent multi-view scene representation. Recent state-of-the-art approaches include low-dimensional generative latent optimization (GLO) vectors [15], learnable affine transformations [21], and bilateral grids (BilaRF) [28]. However, these mitigation strategies face several trade-offs and challenges: 1 arXiv:2601.18336v1 [cs.CV] 26 Jan 2026 • Representation capacity: higher-capacity and lessconstrained modules tend to improve PSNR on the training views but risk modeling more than just photometric variations, often degrading NVS quality. • Interpretability and controllability: the learned parameters are typically non-interpretable (e.g., in GLO or BilaRF), making it difficult to intuitively adjust properties such as brightness or white balance. • Parameters for novel views: since the parameters are optimized independently per frame, it is unclear how to assign appropriate values when synthesizing novel views. The latter is especially challenging due the tendency of these modules to conflate camera sensor intrinsic properties (e.g., vignetting and camera response function) with capture-dependent settings that vary per frame or are adjusted by the ISP (e.g., exposure time and white balance). As a consequence, evaluation protocols commonly assume access to the ground-truth novel view image and estimate a corrective mapping, such as an affine transform, quadratic polynomial, or direct parameter optimization, to minimize the difference between the synthesized and the ground-truth (GT) image before computing the evaluation metrics. But such protocols are inherently flawed as they: (i) deviate from real-world scenarios where GT novel views are unavailable, and (ii) conceal differences between methods by compensating for them through the corrective mapping. To address these challenges, we propose a Physically-Plausible ISP (PPISP) correction module, grounded in the physical principles of camera image formation. Specifically, we disentangle sensor-intrinsic properties and capture-dependent settings through dedicated per-sensor and per-frame modules, respectively, and constrain their effects according to the image formation process (e.g., the exposure module can only modify the overall image brightness). Our model acts as a postprocessing step applied to the raw images rendered from the 3D representation, and enables direct controllability through manual change of the parameters. Moreover, we introduce a PPISP controller that predicts the parameters of the per-frame modules for novel views, analogous to the auto exposure and auto white balance mechanisms in conventional cameras. 2. Related Work Appearance inconsistencies across multi-view input images significantly degrade the quality of radiance field reconstructions and subsequent novel-view synthesis. Such variations are common in unconstrained image collections, for instance when using internet photo collections or captures under uncontrolled lighting conditions. Compensation during reconstruction. To mitigate these inconsistencies, NeRF-W [15] and GS-W [31] introduce learnable per-image latent embeddings (GLO) that are optimized jointly with the scene representation. These latent embeddings enable smooth interpolation within the observed appearance distribution, but may inadvertently get entangled with scene geometry or reflectance when optimized end-to-end. To impose stronger constraints and better align with the image formation process, subsequent works model photometric transformations explicitly. URF [21] represents per-image variations using affine color transformations, while BilaRF [28] extends this idea to per-pixel affine mappings parameterized via bilateral grids. Closest to our approach, ADOP's [22] post-processing models exposure, white balance, camera response function (CRF), and vignetting effects as explicit calibration parameters. However, our formulation better disentangles exposure offset and white balance, while using a more compact CRF model. Recently, Huang et al. [11] and Niemeyer et al. [17] deviate from a frame-based correction and instead learn a 3D exposure neural field, predicting the optimal exposure values for each 3D point. Harmonizing appearance during preprocessing. An alternative strategy is to decouple the compensation from reconstruction and harmonize the input images as a preprocessing step. Shin et al. [24] employ a transformer network to predict bilateral grids that harmonize each image to a chosen reference view. Alzayer et al. [1] instead use a diffusion model to relight images directly, but due to the lack of paired real data, they train their generative model only on synthetic data. To overcome this limitation, Trevithick et al. [27] propose a data generation pipeline that starts from harmonized multi-view inputs and employs a generative model to augment them with diverse lighting conditions. The resulting pseudo-paired dataset enables supervised training of harmonization networks using the original, appearance-consistent images as ground-truth. Novel view synthesis with target appearance. The above methods reconstruct the scene in a canonical or reference appearance, but it remains unclear how to set the parameters of their appearance modules to render an image in a desired target appearance. This target appearance could be user defined or selected to match the appearance that a camera with auto exposure and white balance would produce. This ambiguity poses practical challenges for novel view synthesis and complicates fair evaluation under photometric variation. Prior work typically applies post-render normalization that assumes access to the target image during evaluation: NeRF-W [15] fine-tunes latent embeddings on one half of each image and evaluates on the other, RawNeRF [16] performs channel-wise affine alignment, Mip-NeRF 360 [2] uses a quadratic color basis alignment, and ADOP [22] re-optimizes per-frame parameters. 2 Figure 2. Our proposed pipeline applies a sequence of physically-grounded modules to the input reconstructed radiance (exposure offset, chromatic vignetting, linear color correction and non-linear camera response function). Top: all modules except the controller are jointly optimized during the first training phase. Bottom: the controller is then trained to predict per-frame exposure and color correction for novel views while other modules are frozen. The image sequence shows intermediate outputs after each successive module is applied, illustrating the progressive effects of the pipeline. Such evaluation protocols, however, (i) mask differences between methods and (ii) are infeasible in real-world applications where access to the target image cannot be assumed. In line with the principle that novel views should be rendered solely from reconstructed data without access to target pixels, we introduce a PPISP controller that takes the raw radiance image rendered from the 3D representation as input and outputs the PPISP parameters. We optimize this network on the training views and then directly apply it to the novel views during inference. Somewhat related to our PPISP controller, [18, 26] train a network to predict exposure control for improved feature matching and object detection, respectively. 3. Preliminaries Radiance Field Reconstruction aims to optimize a parametric representation of a scene's volumetric density σ ∈ R and emitted radiance c ∈ R 3 . The radiance L(r) of a camera ray r(x) = o + x d with origin o ∈ R 3 and direction d ∈ R 3 is rendered from this representation as L(r) = Z far near T(x) σ(r(x)) c(r(x)) dt , (1) where T(x) = exp(− R x near σ(r(y)) dy) denotes the transmittance along the ray. The optimization is supervised using ground truth images I captured by one or more cameras with known extrinsics and intrisincs. This standard formulation alone does not account for camera-specific imaging effects. Camera Image Formation is the process through which the radiance L is converted to the final image: I = F(L; Θ) , (2) Here, the function F(·) models the complete image acquisition process, including lens distortions (e.g., vignetting, chromatic aberrations), exposure settings (aperture, shutter time), sensor characteristics (spectral response, noise, gain), and ISP operations according to some parameters Θ. While some components of this process remain constant across acquisition time, others may vary due to manual adjustments or automatic adaptation by the sensor controller. Notation. Let I ∈ R H×W×3 be an RGB image. The color at spatial location u = (i, j) is x = Ii,j ∈ R 3 and its k-th channel value is x = xk = Ii,j,k ∈ R, k ∈ {R, G, B}. Operations defined on channel values or colors are understood element-wise when applied to an image. 4. Method We compensate for photometric inconsistencies across input images by jointly optimizing the scene representation together with a differentiable ISP pipeline that approximates the camera image formation function F(·) defined in Eq. (2). During optimization, this pipeline models both camera-specific and time-varying effects. During inference (i.e., when rendering novel views), the learned controller (Sec. 4.5) predicts the time-varying parameters directly from the radiance L rendered from the scene representation. 3 Our ISP pipeline consists of four sequential modules (see Fig. 2): • Exposure offset accounts for aperture, shutter time and gain variations, • Vignetting models optical attenuation across the sensor, • Color correction models sensor spectral response and white balance adjustments, • Camera response function (CRF) applies a non-linear transformation from sensor irradiance to image colors. Following [8], the first three modules operate linearly on the scene radiance, while the CRF provides the final nonlinear mapping. Fig. 2 shows an overview of the pipeline in the context of the radiance reconstruction and illustrates the individual parts and their effects. 4.1. Exposure Offset We model exposure as a global, per-frame scale on the radiance using a base-2 exponent, mimicking photographic exposure values: I exp = E(L; ∆t) = L2 ∆t , (3) where ∆t ∈ R is an optimizable exposure offset. This offset represents the variation of the radiance intensity reaching the sensor and is specific to the capture. Thus, we estimate one such offset for each frame. 4.2. Vignetting Following Goldman [7], we model per-channel radial intensity falloff using a polynomial in the squared radius around an optimizable optical center: I vig = V(I exp; µ, α) = I exp · v(r; α) , (4) where µ ∈ R 2 is the optical center, α ∈ R 3 are polynomial coefficients, and r = ∥u − µ∥2 is the distance of the pixel location u to the optical center. The attenuation factor v(r) is defined as: v(r) = clip(0,1) 1 + α1 r 2 + α2 r 4 + α3 r 6  . (5) At the start of optimization, we initialize α = 0 and let µ be the image center. Since our vignetting model is chromatic, a falloff polynomial is defined for each color channel by distinct parameter values. 4.3. Color Correction To model effects such as white balance, which may vary per-frame, and gamut differences between multiple cameras, we apply color correction. To disentangle it from exposure correction, we apply a 3 × 3 homography H on RG chromaticities and intensity - following Finlayson et al. [6] - and ensure normalization of the intensity after the transform. Inspired by DeTone et al. [4], we parameterize the color correction as four chromaticity offsets ∆ck, construct H from them, and apply the color correction: I cc = C I vig; {∆ck}k∈{R,G,B,W}  = h(I vig; H) . (6) Let C ∈ R 3×3 denote the RGB→RGI conversion matrix and C−1 its inverse. The intensity normalization can then be defined as : n(x; H) .= xR + xG + xB H · C x 3 + ε . (7) Here, ε is a small constant for numerical stability. This normalization decouples exposure from chromatic correction. The color transform follows compactly as h(x; H) .= C−1 n(x; H) · H · C x . (8) To construct H, we define four 2D source-target chromaticity pairs. Specifically, we fix the source RG chromaticities cs,· to the three primaries and a neutral white: cs,R = (1, 0)T ; cs,G = (0, 1)T ; cs,B = (0, 0)T ; cs,W = 1 3 , 1 3 T , (9) and define the targets ct,· as offsets from these sources ct,k = cs,k + ∆ck for k ∈ {R, G, B, W}. By lifting the 2D chromaticities to homogeneous coordinates and stacking them as S .= [ c˜s,R c˜s,G c˜s,B ] and T .= [ c˜t,R c˜t,G c˜t,B ], we can define M .= [c˜t,W ]× T , (10) where [·]× is the skew-symmetric cross-product matrix. Then, k ∈ R 3 can be obtained via a cross-product of any pair of linearly independent rows i and j, k ∝ mi × mj . (11) where m1, m2, m3 are the rows of M. Finally, we form and normalize H = T diag(k) S −1 , H ← H [H]3,3 . (12) A precise derivation and further details are provided in the Supplementary. 4.4. Camera Response Function Inspired by Grossberg and Nayar [9], we use a piecewise power curve to model non-linear chromatic transformations. The CRF operator G has four learned parameters: I = G(I cc; τ, η, ξ, γ) . (13) 4 Figure 3. Dynamics of the controller module. The predicted exposure offset (inset) depends on the image content of the rendered radiance. Right side: Plot of exposure offsets as predicted for each frame of the caterpillar sequence, with the three displayed frames highlighted. For each channel, the basic S-shaped curve is given by: f0(x; τ, η, ξ) =    a  x ξ τ , 0 ≤ x ≤ ξ , 1 − b  1 − x 1 − ξ η , ξ < x ≤ 1 , (14) setting a and b to match the slope at the inflection point to ensure C 1 continuity: a = η ξ τ (1 − ξ) + η ξ , b = 1 − a . (15) Finally, the CRF image operator G is a composition of this S-curve with a gamma correction: G(x; τ, η, ξ, γ) = f0(x; τ, η, ξ) γ . (16) 4.5. Per-Frame ISP Parameter Controller The exposure offsets and color correction transforms introduced above are valid only for a specific capture, i.e., a single camera pose, and therefore cannot be directly reused for novel view rendering. To address this limitation, we introduce a controller that predicts these parameters from the rendered scene radiance, analogous to how auto exposure and auto white balance works in conventional cameras: (∆t, {∆ck}k∈{R,G,B,W}) = T (L) . (17) Here, T (·) is the camera-specific controller parametric function, which we design as a coarse feature extractor (1×1 convolutions with pooling to a 5×5 grid), followed by a parameter regressor (an MLP with separate output heads). The detailed architecture of the controller is provided in the Supplementary. We optimize the controller in a separate stage once the optimization of the scene representation is complete. At that stage, the underlying reconstruction and all per-camera ISP parameters are frozen, the controller-predicted parameters are applied through the ISP, and the controller itself is trained using the same photometric loss as in the initial phase. A qualitative example of the controller's effects is given in Fig. 3. Optional scalar controls (e.g., exposure compensation or EXIF-derived biases) can be concatenated to the regressor input. 4.6. Regularization Joint optimization of the modules can introduce brightness and color ambiguities between scene radiance and the ISP parameters. To mitigate this, we apply regularization on the previously defined parameters, using the Huber loss Lδ, where δ denotes the threshold. We use superscripts to indicate parameters belonging to specific camera sensors(s) and frames(f) . Brightness. We penalize the mean exposure offset over frames: Lb = λb Lδ=0.1   1 F X F f=1 ∆t (f)   . (18) Color. We penalize the frame-mean of the target chromaticity offsets (element-wise in R 2 ): Lc = λc X k∈{R,G,B,W} Lδ=0.005   1 F X F f=1 ∆c (f) k   . (19) Because chromatic corrections, as done in vignetting and CRF modules, may also introduce localized color shifts, we shrink parameter variance across channels. Let θm,k be the parameters of channel k for module m ∈ {vig, crf}. We penalize their across-channel variance, averaged over parameters: Lvar = λvar X m∈{vig,crf} Vark(θm,k) . (20) 5 Figure 4. Qualitative comparison of novel view synthesis. Row labels indicate datasets and sequences (in italics). Column labels indicate methods. Our method achieves more consistent photometry and better color reproduction across various datasets and sequences. Bottom row: When image metadata such as relative exposure is available, our method can incorporate it to produce a more accurate novel view. Physically-plausible vignetting. For each polynomial, we penalize the center and softly enforce αj ≤ 0: Lvig = λv  ∥µk∥ 2 2 + X j αj 2 +   . (21) Here [x]+ = max(x, 0) is the elementwise rectifier. The overall regularizer is Lreg = Lb + Lc + Lvar + Lvig . (22) 5. Experiments We begin by evaluating the proposed PPISP correction module and controller on standard novel-view synthesis benchmarks, assessing both reconstruction fidelity and novel-view quality (Sec. 5.1). We then demonstrate how our formulation allows us to incorporate image metadata, such as relative exposure, when available (Sec. 5.2). We measure the runtime performance impact (Sec. 5.3). Finally, we analyze the relationship between model capacity, overfitting behavior, and novel-view synthesis performance (Sec. 5.4). Setting. Since the PPISP module as a post-processing operator is reconstruction-agnostic, we integrate it both in 3DGUT [29] and GSplat [30] (an accelerated implementation of 3DGS [12]). Comparison baselines are the post-processing approaches described in BilaRF [28] and ADOP [22]. For experiments, we rely on their reference hyperparameters and reference implementations adapted for 3DGUT and GSplat. To increase the stability of ADOP's method, we increase the strength of their CRF regularization about 100× compared to the reference value. We jointly train the reconstruction method (with the default MCMC configuration) and the post-processing operator for 30k iterations. For the PPISP controller, we freeze both and train the controller for an additional 5k iterations. Metrics. We evaluate the perceptual quality of the rendered views using peak signal-to-noise ratio (PSNR), structural similarity (SSIM), and learned perceptual image patch similarity (LPIPS) metrics. As metrics such as PSNR are highly sensitive to global brightness shifts, and our baselines do not support appearance compensation for novel views, we additionally report metrics computed after affine color alignment, following RawNeRF [16]. We denote this aligned metrics with the suffix "-CC", but emphasize that such comparison masks the differences between the methods and assumes access to the GT target views, which are not available in practice. Datasets. To show the robustness and generality of our method, we conducted experiments on a variety of publicly 6 available datasets: Mip-NeRF 360 [2], Tanks and Temples [14], BilaRF [28], HDR-NeRF [10], and nine static sequences of the Waymo Open Dataset [25]. More details about the scenes, resolution, and training-test splits are available in the Supplementary. To further highlight the differences of the methods in challenging real-world scenarios, we captured a new PPISP dataset consisting of four scenes. Each of them was captured with three different cameras (Apple iPhone 13 Pro, Nikon Z7, and OM System OM-1 Mark II) to ensure variations. Further details of this dataset are available in the Supplementary. 5.1. Novel View Synthesis Benchmark Quantitative results on the standard benchmark scenes are presented in Tab. 1, and qualitative comparisons are shown in Fig. 4. Our method consistently outperforms all baselines across all datasets in terms of PSNR, and for most scenes also in terms of SSIM and LPIPS. Notably, it even surpasses the BilaRF baseline [28] when that baseline is given privileged access to the target image, i.e., when comparing our PSNR against the baseline's PSNR-CC. The relative improvements carry over to the 3DGS [12, 30] integration. The comparison between PSNR and PSNR-CC further highlights the effectiveness of our controller in reproducing the camera's auto-exposure and white-balance behavior. On most datasets, the controller achieves metrics close to those obtained after affine color alignment, indicating that it faithfully predicts the necessary per-frame appearance corrections. The only notable discrepancy appears on the BilaRF dataset, likely due to the fact that this dataset contains some manual settings overrides (indicated by the metadata), which are not captured by our controller. Both PPISP and ADOP [22] employ camera-specific components (vignetting and CRF), which generalize to novel views, leading to improved metrics over BilaRF [28]. Our base image formation model (w/o ctrl.) outperforms both baselines thanks to better separation of concerns of the individual modules and stronger constraints (see also Sec. 5.4). We elaborate on a direct comparison to ADOP in the Supplementary. Our base model still falls short of our full pipeline, which consistently improves novel-view accuracy by providing plausible per-frame parameter estimates via the controller. Ablation. We ablate relative contribution of each module in our pipeline through an ablation study on the TANKS AND TEMPLES dataset. Tab. 2 presents the novel view PSNR when individual components are removed from the full pipeline. The results demonstrate that all modules contribute to the full pipeline's performance, with exposure and vignetting corrections being most critical. 5.2. Using Image Metadata Because our formulation closely mirrors the camera image formation process, it can naturally incorporate image metadata, such as the relative exposure of each frame, whenever available. We demonstrate this capability on the HDR-NeRF [10] and PPISP datasets, both of which use exposure bracketing (i.e., captures with positive and negative exposure compensation) and provide the corresponding metadata. Since the ADOP-style post-processing also models perframe exposure offsets explicitly, we initialize them from known exposure values as proposed in ADOP [22]. For our method, we concatenate the exposure metadata to the input of the controller MLP regressor, allowing it to map rendered radiance plus metadata to effective ISP parameters. Quantitative results in Tab. 3 (PSNR and affine-aligned PSNR) show that supplying calibrated exposure offsets substantially improves novel-view accuracy. Moreover, providing this metadata to the controller yields further gains compared to ADOP, demonstrating our method's ability to leverage metadata for more accurate novel view appearance prediction. 5.3. Runtime Performance Tab. 4 presents the computational performance of the postprocessing methods we evaluated compared to the scene rendering. PPISP (w/o ctrl.) and ADOP [22] have a similar and very small computational footprint (3% of the rendering). The controller is adding a substantial overhead due to the required processing of the input image, but our pipeline remains significantly faster (26% vs 36%) compared to BilaRF on an NVIDIA RTX 5090 GPU. 5.4. ISP Capacity vs. Training and Novel Views Next, we investigate how the capacity of the correction module affect the overfitting (difference between the PSNR on training and novel views) and generalization to novel views. The bilateral grids used in BilaRF [28] provide a highly expressive mechanism for modeling image operations [3] extending beyond simple compensation of photometric inconsistencies. In BilaRF [28], this operation is learned independently for each frame, providing a high modeling capacity. In contrast, our PPISP module intentionally has limited capacity to prevent overfitting, but in turn cannot model complex image operations that mix spatial and intensity effects such as localized tone-mapping. In Tab. 5, we therefore study hybrids of the two approaches. Adding more capacity to per-frame BilaRF [28] with additional per-camera bilateral grids (+PC) does not meaningfully change PSNR on the training views as the model already has sufficient capacity. However, it does slightly improve the generalization as per-camera corrections carry over to novel viewpoints. Increasing our 7 Table 1. Novel view synthesis results across five benchmark datasets. We compare post-processing methods BilaRF [28], ADOP [22], PPISP without controller, and PPISP with controller applied on radiance field reconstruction methods 3DGUT [29] and 3DGS [12, 30]. Metrics with suffix -CC denote color-corrected (affine-aligned) versions that factor out global exposure and color differences. PSNR ↑ PSNR-CC ↑ SSIM ↑ SSIM-CC ↑ LPIPS ↓ LPIPS-CC ↓ BILARF 3DGUT [29] 22.60 23.57 0.804 0.794 0.371 0.371 3DGUT + BilaRF [28] 21.41 25.63 0.764 0.806 0.371 0.344 3DGUT + ADOP [22] 22.95 25.73 0.802 0.799 0.376 0.356 3DGUT + PPISP (w/o ctrl.) 24.08 26.16 0.820 0.825 0.346 0.342 3DGUT + PPISP (w/ ctrl.) 24.12 25.92 0.820 0.816 0.349 0.348 3DGS [12, 30] 23.11 24.59 0.799 0.801 0.367 0.365 3DGS + PPISP (w/ ctrl.) 24.86 26.47 0.824 0.828 0.340 0.337 MIP-NERF 360 3DGUT [29] 27.74 27.65 0.821 0.813 0.262 0.262 3DGUT + BilaRF [28] 24.97 26.64 0.801 0.807 0.260 0.261 3DGUT + ADOP [22] 26.42 27.75 0.815 0.809 0.271 0.265 3DGUT + PPISP (w/o ctrl.) 27.55 28.02 0.819 0.813 0.264 0.264 3DGUT + PPISP (w/ ctrl.) 28.15 28.06 0.821 0.814 0.264 0.264 3DGS [12, 30] 27.69 27.54 0.818 0.809 0.261 0.261 3DGS + PPISP (w/ ctrl.) 27.98 27.89 0.819 0.811 0.260 0.260 TANKS & TEMPLES 3DGUT [29] 22.86 23.46 0.790 0.780 0.312 0.311 3DGUT + BilaRF [28] 19.78 23.46 0.770 0.786 0.298 0.289 3DGUT + ADOP [22] 20.28 24.20 0.769 0.783 0.323 0.303 3DGUT + PPISP (w/o ctrl.) 21.52 24.87 0.783 0.793 0.296 0.290 3DGUT + PPISP (w/ ctrl.) 24.62 25.25 0.809 0.805 0.285 0.284 3DGS [12, 30] 23.03 23.66 0.789 0.781 0.303 0.302 3DGS + PPISP (w/ ctrl.) 24.38 25.16 0.807 0.802 0.281 0.279 WAYMO 3DGUT [29] 25.56 25.21 0.785 0.775 0.397 0.397 3DGUT + BilaRF [28] 21.83 23.66 0.768 0.763 0.397 0.398 3DGUT + ADOP [22] 24.28 25.18 0.781 0.773 0.405 0.400 3DGUT + PPISP (w/o ctrl.) 25.03 25.46 0.786 0.778 0.391 0.391 3DGUT + PPISP (w/ ctrl.) 25.69 25.48 0.787 0.778 0.391 0.392 PPISP-AUTO 3DGUT [29] 22.05 22.20 0.677 0.658 0.453 0.452 3DGUT + BilaRF [28] 20.81 22.30 0.668 0.660 0.440 0.433 3DGUT + ADOP [22] 19.94 22.52 0.670 0.656 0.462 0.441 3DGUT + PPISP (w/o ctrl.) 21.07 23.14 0.677 0.674 0.438 0.434 3DGUT + PPISP (w/ ctrl.) 22.87 23.21 0.687 0.673 0.434 0.433 3DGS [12, 30] 22.29 22.38 0.679 0.662 0.442 0.441 3DGS + PPISP (w/ ctrl.) 22.85 23.17 0.688 0.675 0.426 0.425 Table 2. Component ablation of PPISP on the Tanks and Temples dataset for novel views (NV). Each row shows performance when removing the specified component. NV PSNR ↑ PPISP (full) 24.62 PPISP - no exposure 23.33 PPISP - no vignetting 24.08 PPISP - no color correction 24.27 PPISP - no CRF 24.36 Table 3. Novel View PSNR across datasets with metadata. Our pipeline is able to leverage metadata (e.g. EXIF) from the sensor as a side data provided to the controller regressor. HDR-NERF [10] PPISP metadata PSNR ↑ PSNR-CC ↑ PSNR ↑ PSNR-CC ↑ 3DGUT [29] 17.81 27.37 12.44 18.59 [29] + BilaRF [28] 15.40 26.95 13.39 20.89 [29] + ADOP [22] 15.49 24.14 13.36 17.34 ✓ 31.27 36.10 20.44 21.60 [29] + PPISP 17.86 27.78 14.69 21.19 ✓ 34.30 37.10 21.69 21.94 8 Table 4. Rendering times (ms) on NVIDIA RTX 5090 for the MipNeRF 360 [2] dataset. Time (ms) ↓ % overhead ↓ 3DGUT [29] 3.24 - BilaRF [28] 1.17 36% ADOP 0.10 3% PPISP (w/o ctrl.) 0.10 3% PPISP (w/ ctrl.) 0.84 26% Table 5. Average PSNR on the Tanks and Temples dataset comparing training views (TV) and novel views (NV) for ISP modules with varying capacity. The limited capacity of our proposed pipeline reduces overfitting and leads to better generalization. TV PSNR ↑ NV PSNR ↑ BilaRF + PC 26.83 21.80 PPISP + BilaRF [28] 26.66 23.52 BilaRF [28] 26.87 19.78 ADOP [22] 26.08 20.28 PPISP 25.85 24.62 method's capacity by adding per-frame bilateral grids boosts PSNR on the training views, but noticeably degrades performance on novel views due to overfitting. Overall, our formulation achieves a favorable balance between capacity and generalization to unseen views. 6. Conclusion and Limitations Accurately reconstructing the radiance field of a scene requires accounting for variations in the camera imaging pipeline across the input frames. Ignoring these variations introduces strong biases, leading to spurious color shifts and geometric artifacts. In this work, we introduced a differentiable post-processing pipeline whose design permits to simulate the imaging process while remaining highly constrained to prevent reconstruction bias. We further proposed a controller that improves generalization to novel views by predicting per-frame imaging parameters directly from the rendered radiance. Limitations. Our method shows superior generalization to novel views (Tab. 1), but it sometimes struggles to match the baselines on the training-views (Tab. 5). This can be partially accredited to overfitting, but our formulation also ignores some important optical effects such as localized tone-mapping commonly found in modern phone cameras; lens flares, which are prominent in night scenes; and similar spatially-adaptive effects. While the proposed controller enables generalization to novel views, its ability to infer exposure and color-correction parameters from rendered radiance depends on the existence of meaningful correlations in the data. When such correlations are absent, for example when the physical camera controls (e.g., shutter, aperture, ISO) are manually overridden, the controller must rely on extra metadata to predict correct values. 7. Acknowledgements We thank our colleagues Qi Wu, Janick Martinez Esturo, Andras B ´ odis-Szomor ´ u, and Nick Schneider for their sug- ´ gestions, feedback, and valuable discussions. References [1] Hadi Alzayer, Philipp Henzler, Jonathan T. Barron, Jia-Bin Huang, Pratul P. Srinivasan, and Dor Verbin. Generative multiview relighting for 3d reconstruction under extreme illumination variation. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 10933- 10942, 2025. 2 [2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. In CVPR, 2022. 2, 7, 9, 6 [3] Jiawen Chen, Andrew Adams, Neal Wadhwa, and Samuel W Hasinoff. Bilateral guided upsampling. ACM Transactions on Graphics (TOG), 35(6):1-8, 2016. 7 [4] Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. Deep image homography estimation, 2016. Preprint. 4 [5] Daniel Duckworth, Peter Hedman, Christian Reiser, Peter Zhizhin, Jean-Franc¸ois Thibert, Mario Luciˇ c, Richard ´ Szeliski, and Jonathan T. Barron. Smerf: Streamable memory efficient radiance fields for real-time large-scene exploration, 2023. 2 [6] Graham Finlayson, Han Gong, and Robert B. Fisher. Color homography: theory and applications. IEEE TPAMI, 41(1): 20-33, 2017. 4 [7] Daniel B Goldman. Vignette and exposure calibration and compensation. IEEE transactions on pattern analysis and machine intelligence, 32(12):2276-2288, 2010. 4 [8] Michael D. Grossberg and Shree K. Nayar. Determining the camera response from images: What is knowable? IEEE TPAMI, 25(11):1455-1467, 2003. 4, 2 [9] Michael D. Grossberg and Shree K. Nayar. Modeling the space of camera response functions. IEEE TPAMI, 26(10): 1272-1282, 2004. 4 [10] Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, Xuan Wang, and Qing Wang. Hdr-nerf: High dynamic range neural radiance fields. In CVPR, pages 18398-18408, 2022. 7, 8, 2, 5, 6 [11] Xin Huang, Qi Zhang, Ying Feng, Hongdong Li, and Qing Wang. Ltm-nerf: Embedding 3d local tone mapping in hdr neural radiance field. IEEE TPAMI, 2024. 2 [12] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkuhler, ¨ and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, 42 (4), 2023. 6, 7, 8 9 [13] Agnan Kessy, Alex Lewin, and Korbinian Strimmer. Optimal whitening and decorrelation. The American Statistician, 72 (4):309-314, 2018. 4 [14] Arno Knapitsch, Jaesik Park, Qian-Yi Zhou, and Vladlen Koltun. Tanks and temples: Benchmarking large-scale scene reconstruction. ACM Transactions on Graphics, 36(4), 2017. 7, 5, 6 [15] Ricardo Martin-Brualla, Noha Radwan, Mehdi S. M. Sajjadi, Jonathan T. Barron, Alexey Dosovitskiy, and Daniel Duckworth. Nerf in the wild: Neural radiance fields for unconstrained photo collections. In CVPR, pages 7210-7219, 2021. 1, 2 [16] Ben Mildenhall, Peter Hedman, Ricardo Martin-Brualla, Pratul P. Srinivasan, and Jonathan T. Barron. NeRF in the dark: High dynamic range view synthesis from noisy raw images. In CVPR, 2022. 2, 6 [17] Michael Niemeyer, Fabian Manhardt, Marie-Julie Rakotosaona, Michael Oechsle, Christina Tsalicoglou, Keisuke Tateno, Jonathan T. Barron, and Federico Tombari. Learning neural exposure fields for view synthesis. In NeurIPS, 2025. to appear. 2 [18] Emmanuel Onzon, Fahim Mannan, and Felix Heide. Neural auto-exposure for high-dynamic range object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021. 3 [19] Linfei Pan, Daniel Barath, Marc Pollefeys, and Johannes Lutz Schonberger. Global Structure-from-Motion ¨ Revisited. In ECCV, 2024. 6 [20] Keunhong Park, Philipp Henzler, Ben Mildenhall, Jonathan T. Barron, and Ricardo Martin-Brualla. Camp: Camera preconditioning for neural radiance fields. ACM TOG, 42(6):1-11, 2023. 4 [21] Konstantinos Rematas, Andrew Liu, Pratul P. Srinivasan, Jonathan T. Barron, Andrea Tagliasacchi, Tom Funkhouser, and Vittorio Ferrari. Urban radiance fields. CVPR, 2022. 1, 2 [22] Darius Ruckert, Linus Franke, and Marc Stamminger. Adop: ¨ Approximate differentiable one-pixel point rendering. ACM TOG, 41(4):99:1-99:14, 2022. 2, 6, 7, 8, 9, 1, 3, 4, 5 [23] Johannes Lutz Schonberger and Jan-Michael Frahm. ¨ Structure-from-motion revisited. In CVPR, 2016. 6 [24] Jisu Shin, Richard Shaw, Seunghyun Shin, Zhensong Zhang, Hae-Gon Jeon, and Eduardo Perez-Pellitero. Chroma: Consistent harmonization of multi-view appearance via bilateral grid prediction, 2025. Preprint. 2 [25] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR, 2020. 7, 5, 6 [26] Justin Tomasi, Brandon Wagstaff, Steven L Waslander, and Jonathan Kelly. Learned camera gain and exposure control for improved visual feature detection and matching. IEEE Robotics and Automation Letters, 6(2):2028-2035, 2021. 3 [27] Alex Trevithick, Roni Paiss, Philipp Henzler, Dor Verbin, Rundi Wu, Hadi Alzayer, Ruiqi Gao, Ben Poole, Jonathan T. Barron, Aleksander Holynski, Ravi Ramamoorthi, and Pratul P. Srinivasan. Simvs: Simulating world inconsistencies for robust view synthesis. arXiv, 2024. 2 [28] Yuehao Wang, Chaoyi Wang, Bingchen Gong, and Tianfan Xue. Bilateral guided radiance field processing. ACM TOG, 43(4):148:1-148:13, 2024. 1, 2, 6, 7, 8, 9, 3, 5 [29] Qi Wu, Janick Martinez Esturo, Ashkan Mirzaei, Nicolas Moenne-Loccoz, and Zan Gojcic. 3dgut: Enabling distorted cameras and secondary rays in gaussian splatting. In CVPR, 2025. 6, 8, 9, 3 [30] Vickie Ye, Ruilong Li, Justin Kerr, Matias Turkulainen, Brent Yi, Zhuoyang Pan, Otto Seiskari, Jianbo Ye, Jeffrey Hu, Matthew Tancik, and Angjoo Kanazawa. gsplat: An open-source library for gaussian splatting. Journal of Machine Learning Research, 26(34):1-17, 2025. 6, 7, 8 [31] Dongbin Zhang, Chuming Wang, Weitao Wang, Peihao Li, Minghan Qin, and Haoqian Wang. Gaussian in the wild: 3d gaussian splatting for unconstrained image collections. In European Conference on Computer Vision, pages 341-359. Springer, 2024. 2 10 PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction Supplementary Material This supplementary material provides additional experiments, method details, and implementation specifications to complement the main paper. Sec. A presents extended experimental results, including a detailed comparisons with ADOP's image formation model [22] and additional experiments on components (camera calibration and exposure identifiability). Sec. B provides further method details, i.e., mathematical derivations of our color correction formulation and specifications of our per-frame controller architecture. Sec. C details optimization settings, regularization weights, learning rate schedules, and dataset specifications used throughout our experiments. Finally, Sec. D discusses interactive manual control capabilities of our method. A. Additional Experiments To complete the main paper experiments, we provide further qualitative results in Fig. 5 and present the detail of the novel-view PSNR for every scene in Tab. 6. A.1. Detailed Comparison with ADOP [22] In the related work (Sec. 2), we mention that ADOP [22] implements a similar image formation model as ours. We deviate in the color correction and CRF. Here, we provide a detailed comparison, expanding on the main results in Sec. 5. White balance and exposure decoupling. In Sec. 4.3, we claim that our color correction method, which operates on 2D chromaticities instead of 3D color and normalizes the intensity post-transformation, decouples the white balance from the exposure correction. We evaluate this by computing the Pearson correlation coefficient (PCC) between the estimated exposure offset and the white point offset, ∆cW , which controls the white balance and compare our method against ADOP's which uses per-channel white-point gains. The PCC is defined as: rX,Y = cov(X, Y ) σXσY (23) where rX,Y is the Pearson correlation coefficient between variables X and Y , cov(X, Y ) is the covariance between X and Y , and σX and σY are the standard deviations of X and Y , respectively. A PCC near 1 indicates strong linear correlation, and a PCC near 0 indicates weak or no correlation. A representative result is shown in Fig. 6. We find that the PCC numbers for our method are substantially lower as compared to ADOP's method on all sequences, indicating an improved decoupling of white balance and exposure correction. Figure 7 further highlights the importance of decoupling color and exposure corrections: When exposure and color are coupled, the CRF will also be entangled in order to compensate for the value-dependent color shift. That, in turn, hinders the controllability of both aspects since neither can be changed without also affecting the other. CRF stability in challenging sequences. In Sec. 4.4, we provide a formulation for the camera response function that is constrained to be monotonically increasing and smooth by design. This ensures that the optimization remains stable. In some sequences, particularly when large photometric variations were present, we found that this offers an improvement over ADOP's [22] CRF formulation, which uses 25 discrete nodes which are interpolated linearly and requires a smoothness loss. A degenerate case of ADOP's CRF is illustrated in Fig. 7 (third row), where the learned green and red channels of the CRF are split into lower and upper sections with a reversal. This violates the assumption that the CRF is monotonically increasing. While the postprocessed image still remains close in brightness and color to the actual scene due to corrections being self-consistent, it falls apart with strong color artifacts when applying a controlled exposure offset. A.2. Online Camera Calibration Since certain parts of the PPISP pipeline, namely the vignetting (Sec. 4.2) and CRF (Sec. 4.4), are shared across all frames of a camera, the process of jointly optimizing them with the radiance field reconstruction can be understood as an online camera calibration. We compared the recovered per-camera parameters across multiple sequences qualitatively in Fig. 8, where multiple plots are overlaid. Same color implies same dataset. The close overlap of the curves from the same datasets and the distinct shapes between datasets indicate that our method can robustly extract these calibrations. It also suggests that the camera-specific curves are disentangled from scene radiance and other corrective effects, otherwise we would expect an ambiguous mixing of them. A.3. Identifiability of Exposure Offsets In Sec. 5.2, we tested the effectiveness of using image exposure metadata to guide the image formation process. Here, we consider the inverse problem of identifying calibrated 1 Figure 5. Qualitative comparison of novel view synthesis, additional examples. Row labels indicate datasets and sequences (in italics). Column labels indicate methods. 0.2 0.1 0.0 0.1 t 0.2 0.1 0.0 0.1 White Balance rR, t = 0.911 rB, t = 0.831 0.5 0.0 0.5 t 0.002 0.001 0.000 0.001 0.002 rR, t = -0.249 rB, t = -0.230 Figure 6. Correlation between optimized exposure offset and white balancing variables in SMERF's [5] alameda sequence. Left: ADOP's [22] red and blue channel scaling. Right: The offsets of the white point of our homography-based correction. The Pearson correlation coefficient for each component is inset. exposure offsets. In this experiment, per-frame exposure offsets are freely optimized and compared against the relative exposure metadata present in the HDR-NeRF [10] and PPISP datasets. According to Grossberg and Nayar [8], there is an "exponential ambiguity", which states that transforming both the inverse of the CRF and the radiance by some power produces exactly the same image intensities. Since our exposure offsets are parameterized in log-space, applying a power to the radiance corresponds to a scaling in parameter space. Thus, for this experiment, we apply an optimal affine transform on the recovered exposure offsets and compute the error on the transformed data. As illustrated in Fig. 9 for a representative sequence, calibrated exposure metadata is matched closely. B. Additional Method Details B.1. Color Correction In Sec. 4.3, we propose a color correction method based on a 3 × 3 homography matrix H, applied on RG chromaticities and intensity, followed by an intensity normalization. For the parameterization of H, we show a construction from chromaticity offsets ∆ck that control the mapping from source to target chromaticities. In this section, we provide a more detailed derivation. Furthermore, we detail the preconditioning we apply to the chromaticity offsets ∆ck. Derivation and equivalence to direct linear transformation. We derive the construction of H in detail and show that the resulting matrix is equivalent to the standard method for constructing homography matrices from sourcetarget pairs, the direct linear transformation (DLT). In Sec. 4.3, we define source and target chromaticity vector pairs c{s,t},{R,G,B,W}. The homogeneous lifts of these vectors are denoted with a tilde, c˜{s,t},{R,G,B,W}. The S and T matrices are built by stacking the lifted source and target red, green, and blue chromaticity vectors, respectively. We note that S is constant and has an inverse S −1 . Reduction using three correspondences. By definition, a homography is a colinear transformation (collineation), i.e., transformed vectors are identical to the original ones up to scale: H c˜s,i ∼ c˜t,i for i ∈ {R, G, B}. Using the stacked matrices S and T, it follows that there exist nonzero 2 Table 6. Per-scene novel view PSNR comparison. We compare post-processing methods applied on top of 3DGUT reconstruction across all sequences. Higher is better (↑). Dataset Scene 3DGUT [29] + BilaRF [28] + ADOP [22] + PPISP (w/o ctrl.) + PPISP (w/ ctrl.) BILARF building 24.85 22.81 25.30 26.36 26.46 chinesearch 18.34 20.44 21.27 22.13 21.62 lionpavilion 24.16 24.11 22.89 25.06 24.76 nighttimepond 27.11 21.54 25.07 27.68 28.16 pondbike 25.28 21.17 24.96 26.33 26.04 statue 22.40 21.01 22.84 22.84 22.26 strat 16.06 18.76 18.34 18.17 19.55 MIP-NERF 360 bicycle 25.28 24.26 24.54 24.95 25.72 bonsai 32.52 28.57 30.33 32.10 33.02 counter 29.36 26.30 27.58 28.89 29.50 flowers 21.80 20.10 21.54 21.76 21.95 garden 26.85 24.06 26.10 27.14 27.31 kitchen 31.86 27.50 28.08 30.51 32.14 room 32.11 29.53 30.76 32.95 32.84 stump 26.90 24.90 26.59 27.03 27.28 treehill 22.97 19.46 22.25 22.59 23.55 TANKS AND TEMPLES caterpillar 22.61 19.19 18.15 19.74 25.18 ignatius 22.03 20.01 20.47 20.77 24.04 train 22.06 19.04 18.95 20.17 23.74 truck 24.72 20.88 23.56 25.38 25.51 WAYMO 10275144660749673822 5755 561 5775 561 24.73 20.59 23.68 24.30 25.17 1265122081809781363 2879 530 2899 530 28.39 24.47 26.30 27.50 28.31 15959580576639476066 5087 580 5107 580 27.52 24.06 26.54 27.04 27.77 16470190748368943792 4369 490 4389 490 23.82 20.17 22.09 23.69 24.21 16608525782988721413 100 000 120 000 23.29 19.86 22.62 22.91 23.27 16646360389507147817 3320 000 3340 000 26.65 23.71 24.84 25.86 26.48 17244566492658384963 2540 000 2560 000 27.25 22.19 26.00 26.31 27.39 1999080374382764042 7094 100 7114 100 24.10 20.85 23.18 23.65 24.34 744006317457557752 2080 000 2100 000 24.26 20.53 23.31 24.05 24.30 PPISP-AUTO huerstholz auto 19.23 18.76 18.88 19.24 19.81 struktur28 auto 24.21 22.80 21.97 22.25 25.28 toro auto 22.24 20.56 18.44 20.20 23.01 valiant auto 22.51 21.14 20.47 22.58 23.39 k = (kR, kG, kB) ⊤ such that H S = T diag(k) =⇒ H(k) = T diag(k) S −1 . (24) Thus, the homography is reduced to three column scales up to a common factor. Fourth correspondence via colinearity. To find k, we write the source white point as c˜s,W = S b with barycentric b = ( 1 3 , 1 3 , 1 3 ) ⊤ . We require H c˜s,W ∼ c˜t,W . Another way to express this colinearity constraint is c˜t,W × T diag(b) k  = 0 . Using the skew-symmetric matrix [·]× with [x]×y = x×y, this yields the homogeneous linear system [c˜t,W ]× T diag(b) k = 0 . For the white point, diag(b) ∝ I, so the constraint reduces to the 3 × 3 system M k = 0 with M = [c˜t,W ]× T . Generically rank(M) = 2, so the right nullspace is 1D and determines k up to scale. A practical closed form is to take any cross of two independent rows ri , rj of M, i.e.: k ∝ ri × rj . Substituting k into H(k) and normalizing by an arbitrary scalar (e.g., set [H]3,3 = 1) gives the desired homography. 3 Figure 7. Comparison of ADOP [22]-style post-processing including exposure control against our method. Row labels indicate the postprocessing method and the sequence name (in italics). The CRF for ADOP's formulation compensates for the color artifacts baked into the radiance field only at a specific exposure value. But when controlling exposure for novel views, color artifacts are exacerbated. In contrast, both our method's radiance field and output remain neutral since all corrections are decoupled. Equivalence to the 4-point DLT. The classical DLT stacks the four constraints into A h = 0 for the 9-vector h of H (up to scale), and solves for the 1D right-nullspace of A. Our construction enforces the same constraints factorized through the invertible S: three correspondences reduce to the column scales k, and the fourth yields M k = 0. Under non-degenerate configurations (i.e., the columns of T are not colinear and rank(M) = 2), both methods recover the same H up to an overall scalar. Degeneracies and identity case. If rank(T) < 2 or rank(M) < 2, k is ill-defined, mirroring DLT degeneracies. When targets equal sources, T = S, c˜t,W = c˜s,W , and k ∝ (1, 1, 1), yielding H proportional to the identity after normalization. Preconditioning of the chromaticity offsets. Our color correction method involves a conversion from RGB color to RGI (red-green chromaticity and intensity) and back, with I = R + G + B and B = I − R − G in terms of components. In our optimization setting, this correlates the gradients of the individual chromaticity offsets {∆ci} with the blue channel. In addition to that, the output image is generally more sensitive to changes in the white point than an offset in the RGB primaries. In order to whiten the color correction and decorrelate the individual components, we apply ZCA preconditioning with proxy Jacobians following [13, 20]. We precondition the 8-dimensional vector of chromaticity offsets {∆ci}i∈{R,G,B,W}. We use a block decomposition into four 2 × 2 blocks (one per control point) in place of the full 8 × 8 transform. B.2. Controller Architecture The overall architecture of the per-frame ISP controller is given in Sec. 4.5. Here, we provide the complete architectural specifications. Input and output. The controller takes as input the rendered scene radiance L ∈ R H×W×3 . Extra inputs, such as image metadata, are input at the beginning of the parameter regression stage. The controller outputs 9 parameters: an exposure offset ∆t ∈ R and eight color correction offsets {∆ci}i∈{R,G,B,W}. Feature extraction stage. The feature extractor processes the input radiance using a sequence of 1x1 convolutions and pooling operations. 4 0.0 0.2 0.4 0.6 0.8 1.0 Input intensity 0.0 0.2 0.4 0.6 0.8 1.0 Output intensity CRF HDR-NeRF T&T Waymo 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 Distance to center 0.4 0.6 0.8 1.0 Attenuation factor Vignetting HDR-NeRF T&T Waymo Figure 8. Recovered camera-specific parameters across datasets. Top: The calibrated CRF of three sequences of each of the HDRNeRF [10], Tanks and Temples [14], and Waymo Open Drive [25] dataset are overlaid. Bottom: For the same sequences and datasets, the vignetting falloff curves are compared. 0 200 400 600 Frame index 2 1 0 1 2 3 t RMSE = 0.1427 Figure 9. Optimized exposure parameters per frame and given exposure metadata for the huerstholz sequence in the PPISP dataset. Colors indicate individual cameras. First, a 1x1 convolution maps the 3-channel input to 16 feature channels. This is followed by max pooling with a factor of 3 in each spatial dimension, reducing the resolution to H/3 × W/3. A ReLU activation is then applied. Next, a second 1x1 convolution expands the features to 32 channels, followed by ReLU. A third 1x1 convolution produces 64 feature channels, yielding a feature map F ∈ R H/3×W/3×64 . Then, spatial aggregation is performed. An adaptive average pooling operation reduces the spatial dimensions to a 5 × 5 grid, producing a coarse feature representation Fpool ∈ R 5×5×64. This grid captures multi-scale spatial statistics of the scene while maintaining spatial locality, analogous to metering zones in conventional cameras. Parameter regression stage. The pooled features are flattened into a 1600-dimensional vector (5 × 5 × 64). If available, image metadata may be concatenated at this stage. This is input into an MLP with three hidden layers, each containing 128 neurons with ReLU activations. The output consists of two parallel linear heads: one producing the exposure offset and the other producing the 8 color correction parameters. C. Additional Experiment Details We provide optimization hyperparameters, regularization weights, and dataset specifications used throughout our experiments. C.1. Optimization settings Regularization weights. In Sec. 4.6, we specify the regularizer terms that break brightness and color ambiguities and ensure physically-plausible vignetting. In Tab. 7, we detail the numerical values used for each λ term. Table 7. Regularization coefficients. Term λ λb 1.0 λc 1.0 λvar 0.1 λv 0.01 Optimizer, learning rates, and schedules. For all postprocessing modules including BilaRF [28], ADOP's formulation [22], and our method, we use the Adam optimizer. We use the following learning rate scheduling with an initial delay (zero learning rate), linear warmup, and exponential decay. lr(s) =    0, s < sd, lr0  fs + (1 − fs) s − sd sw  , sd ≤ s < sd + sw, lr0  f 1/smax f s−sd−sw , s ≥ sd + sw. (25) Where: • lr0 - base learning rate. • s - current training step. • sd - delay steps (learning rate held at zero). • sw - warmup steps (linear ramp from fslr0 to lr0). • smax - number of decay steps. 5 Figure 10. Our low-parametric formulation of the different image processing steps enables manual editing. Top left shows the input image. Other images have details overlaid, such as the primary effect being applied and an abstract visualization. In the color correction examples, the white dots correspond to the four target chromaticities ct,{R,G,B,W}, which can be intuitively manipulated. • fs - start factor for warmup (e.g., 0.01). • ff - final factor reached after decay (e.g., 0.01). Tab. 8 details the values used during experiments. Table 8. Learning rate scheduler hyperparameters. Term Value lr0 0.002 sd 0 sw 500 fs 0.01 smax 30000 ff 0.01 In Sec. 5.4, we experiment with combined post-processing methods. In these cases, the BilaRF module as combined with PPISP and per-camera bilateral grids use sd = 5000 and sw = 1000 with otherwise the same hyperparameters as in Tab. 8. C.2. Datasets In Sec. 5, we outline the datasets used for experiments. In this section, we define the datasets in more detail. Specific choice of sequences. We chose the following sequences from each dataset: • Mip-NeRF 360 [2]: All nine sequences, • Tanks and Temples [14]: Four sequences, namely train, truck, caterpillar, and ignatius, • BilaRF [28]: All seven sequences, • HDR-NeRF [10]: All four real-camera sequences, • Waymo Open Dataset [25]: Nine mostly static sequences, explicitly listed in Tab. 9; All five cameras used. PPISP dataset details. As stated in Sec. 5, we captured our own dataset using three cameras, including two modern Table 9. Waymo Open Dataset [25] sequence names. Sequence Name 74400631745755752 2080 000 2100 000 126512208180978136 2879 530 2899 530 199908037438276404 7094 100 7114 100 102751446607496738 5755 561 5775 561 159595805766394760 5087 580 5107 580 164701907483689437 4369 490 4389 490 166085257829887214 100 000 120 000 166463603895071478 3320 000 3340 000 172445664926583849 2540 000 2560 000 mirrorless and a smartphone camera. We provide further context here. For all cameras and scenes, we used exposure bracketing of ±2 EV to capture HDR data. The aperture and focus were set manually and remained fixed. Image stabilization was disabled. Each scene was captured in raw format. The raw photos were developed with NX Studio and OM Workspace for the Nikon and OM System photos, and Adobe Lightroom Classic for the iPhone photos, respectively. A color calibration target placed in the scene was used to white balance. For each scene, we additionally picked certain exposures out of the brackets and re-developed them with normalized, automatic exposure compensation and white balancing, creating a more challenging setting for the controller module. We denote this derived dataset PPISP-auto. Pre-processing. For all datasets including our own, where camera poses or sparse point clouds were not originally available, we processed them through COLMAP [23] and GLOMAP [19] to produce the necessary inputs for the radiance field reconstruction. 6 We used downsampled versions of the original camera images so that the maximum effective side length of each input image did not exceed 2000 pixels. E.g., for Mip-NeRF 360's [2] garden sequence, we used 4× downsampling, and for bonsai, we used 2×. We used a seven to one split of test views to validation views for evaluation throughout. D. Manual Control Our parametric ISP formulation enables intuitive manual editing and artistic control. Fig. 10 demonstrates various edits applied to a reconstructed scene, including adjustments to exposure, white balance, vignetting, and camera response. The low-dimensional and disentangled representation ensures meaningful and predictable edits, facilitating interactive workflows for applications such as artistic rendering, temporal consistency enforcement, or selective photometric matching. 7]

# Source Code Context

Generated on: 2026-02-26T11:05:44+05:30

## Repository Overview
- Total Files: 14
- Total Size: 200199 bytes

## Directory Structure
```
README.md
ppisp/
  __init__.py
  bindings.h
  ext.cpp
  report.py
  src/
    ppisp_impl.cu
    ppisp_math.cuh
    ppisp_math_bwd.cuh
setup.py
tests/
  README.md
  __init__.py
  test_cuda_vs_torch.py
  test_gradcheck.py
  torch_reference.py

```

## File Contents


### File: README.md

```markdown
# PPISP

Code for the work "PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction".

[Project website](https://research.nvidia.com/labs/sil/projects/ppisp/) | [Paper](https://arxiv.org/abs/2601.18336)

## Overview

PPISP is a learned post-processing module for radiance field rendering that models common photometric variations found in real-world multi-camera video captures:

- **Exposure compensation** (per-frame): Corrects brightness variations caused by auto-exposure or changing lighting conditions
- **Vignetting** (per-camera): Models radial light falloff from lens optics
- **Color correction** (per-frame): Handles white balance drift and color cast variations via chromaticity homography
- **Camera Response Function (CRF)** (per-camera): Learns the nonlinear tone mapping applied by each camera's ISP

The module also includes a **controller** that predicts per-frame exposure and color corrections from rendered radiance images. This enables consistent novel view synthesis without requiring per-frame parameters at inference time.

All processing is implemented as a differentiable CUDA kernel for efficient integration into training pipelines.

## Adding PPISP as a Dependency

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use. See [ATTRIBUTIONS.md](ATTRIBUTIONS.md) for a list of dependencies and their licenses.

### Build System

PPISP uses a two-file build configuration:

- **`pyproject.toml`** declares build-time requirements (`setuptools`, `wheel`) and runtime dependencies (`torch`)
- **`setup.py`** compiles the CUDA extension using `torch.utils.cpp_extension`, which requires PyTorch to be available during the build process

To compile the CUDA extension against the exact PyTorch version in your environment for ABI compatibility, we recommend installing with `--no-build-isolation`.

### Installation

**Local development install:**

```bash
pip install . --no-build-isolation
```

**As a dependency in `requirements.txt`:**

Add the following to your `requirements.txt`, optionally pinning a specific version:

```
ppisp @ git+https://github.com/nv-tlabs/ppisp.git@v1.0.0
```

Then install with the `--no-build-isolation` flag:

```bash
pip install -r requirements.txt --no-build-isolation
```

## Integration into Existing Radiance Field Reconstruction Pipelines

### Basic Usage

```python
from ppisp import PPISP, PPISPConfig

# 1. Initialize
ppisp = PPISP(num_cameras=3, num_frames=500)

# 2. Create optimizers and scheduler
ppisp_optimizers = ppisp.create_optimizers()
ppisp_schedulers = ppisp.create_schedulers(ppisp_optimizers, max_optimization_iters)

# 3. Training loop
for step in range(max_optimization_iters):
    # ...

    # Render raw RGB from your radiance field
    rgb_raw = renderer(camera_idx, frame_idx)  # [..., 3]

    # Apply PPISP post-processing
    rgb_out = ppisp(
        rgb_raw,
        pixel_coords,           # [..., 2]
        resolution=(W, H),
        camera_idx=camera_idx,
        frame_idx=frame_idx,
    )

    # Add PPISP regularization loss to other losses
    loss = reconstruction_loss(rgb_out, rgb_gt) + ppisp.get_regularization_loss()
    loss.backward()

    # Step optimizers and scheduler
    for opt in ppisp_optimizers:
        opt.step()
        opt.zero_grad(set_to_none=True)
    for sched in ppisp_schedulers:
        sched.step()

# 4. Novel view rendering: pass camera_idx as usual, frame_idx=-1
rgb_out = ppisp(rgb_raw, pixel_coords, resolution=(W, H), camera_idx=camera_idx, frame_idx=-1)
```

Use `PPISPConfig` to customize regularization weights, learning rates, and controller activation timing.

### Controller Distillation Mode

Controller distillation is enabled by default. When the controller activates, the scene representation (e.g., Gaussians, NeRF) should be frozen. This allows the controller to learn from fixed PPISP parameters without interference from scene updates:

```python
from ppisp import PPISP

ppisp = PPISP(num_cameras=3, num_frames=500)

# Training loop: freeze scene parameters when controller activates (default: 80% of training)
controller_activation_step = int(0.8 * max_optimization_iters)

for step in range(max_optimization_iters):
    if step >= controller_activation_step:
        freeze_scene_parameters()  # Your function to freeze Gaussians/NeRF params
    # ... rest of training loop ...
```

To disable distillation, set `controller_distillation=False` in the config.

When distillation is enabled and the controller activates:
- PPISP automatically freezes its internal parameters (exposure, vignetting, color, CRF)
- PPISP detaches the input RGB to prevent gradients flowing back to the scene
- Only the controller network receives gradients

### Exporting PDF Reports (Optional)

After training, you can export a PDF summary of the learned PPISP parameters:

```python
from pathlib import Path
from ppisp.report import export_ppisp_report

# Export PDF reports (one per camera) and a JSON with all parameters
pdf_paths = export_ppisp_report(
    ppisp,
    frames_per_camera=[100, 100, 100],  # frames per camera
    output_dir=Path("./ppisp_reports"),
    camera_names=["cam_front", "cam_left", "cam_right"],  # optional
)
```

The report includes visualizations for exposure, vignetting, color correction, and CRF per camera.

## BibTeX

```bibtex
@misc{deutsch2026ppispphysicallyplausiblecompensationcontrol,
      title={PPISP: Physically-Plausible Compensation and Control of Photometric Variations in Radiance Field Reconstruction},
      author={Isaac Deutsch and Nicolas Moënne-Loccoz and Gavriel State and Zan Gojcic},
      year={2026},
      eprint={2601.18336},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.18336},
}
```

```



### File: ppisp\__init__.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPISP: Physically-Plausible Image Signal Processing

A learned post-processing module for radiance field rendering that models:
- Exposure compensation (per-frame)
- Vignetting (per-camera)
- Color correction (per-frame)
- Camera Response Function / tone mapping (per-camera)

Includes a controller for predicting exposure and color
parameters from rendered radiance images.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

import ppisp_cuda as _C


# =============================================================================
# Version
# =============================================================================

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ppisp")
except PackageNotFoundError:
    __version__ = "0.0.0"  # fallback for editable/dev install

# =============================================================================
# Constants
# =============================================================================

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
# Stored as 8x8 block-diagonal matrix for efficient single-matmul application
# Used by PPISP.get_regularization_loss() for color mean regularization
_COLOR_PINV_BLOCK_DIAG = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),  # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),  # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),  # Green
    torch.tensor([[0.0128369, -0.0034654],
                 [-0.0034654, 0.0128158]]),  # Neutral
).to(torch.float32)

# Constants (match config.h)
NUM_VIGNETTING_ALPHA_TERMS = 3


def _normalize_index(idx: torch.Tensor | int | None, name: str) -> int:
    """Normalize camera/frame index to int.

    Args:
        idx: Index as tensor (numel==1), int, or None
        name: Parameter name for error messages

    Returns:
        -1 if None, otherwise the integer value
    """
    if idx is None:
        return -1
    if isinstance(idx, torch.Tensor):
        assert idx.numel() == 1, (
            f"{name} tensor must have exactly 1 element (batch_size==1), "
            f"got {idx.numel()}"
        )
        return int(idx.item())
    return int(idx)


COLOR_PARAMS_PER_FRAME = 8
CRF_PARAMS_PER_CHANNEL = 4
VIGNETTING_PARAMS_PER_CHANNEL = 2 + NUM_VIGNETTING_ALPHA_TERMS


# =============================================================================
# PPISP Configuration
# =============================================================================

@dataclass
class PPISPConfig:
    """Configuration for PPISP module.

    Includes regularization weights, optimizer settings, scheduler settings,
    and controller activation settings.
    All defaults are tuned for typical radiance field training scenarios.
    """

    # Controller settings
    use_controller: bool = True
    """Enable the controller for predicting per-frame corrections.
    When False, zero corrections are used for novel views.
    """

    controller_distillation: bool = True
    """Use distillation to train the controller.
    When True, the controller is trained with freezing the other PPISP parameters and detaching
    the input so that gradients only flow through the controller.
    """

    controller_activation_ratio: float = 0.8
    """Relative training step at which to activate the controller.
    Controller activates when step >= controller_activation_ratio * max_optimization_iters.
    Default 0.8 means the controller activates at 80% of training.
    """

    # Regularization weights
    exposure_mean: float = 1.0
    """Encourage exposure mean ~ 0 to resolve SH <-> exposure ambiguity."""

    vig_center: float = 0.02
    """Encourage vignetting optical center near image center."""

    vig_channel: float = 0.1
    """Encourage similar vignetting across RGB channels."""

    vig_non_pos: float = 0.01
    """Penalize positive vignetting alpha coefficients (should be <= 0)."""

    color_mean: float = 1.0
    """Encourage color correction mean ~ 0 across frames."""

    crf_channel: float = 0.1
    """Encourage similar CRF parameters across RGB channels."""

    # Optimizer settings for PPISP main params
    ppisp_lr: float = 0.002
    """Learning rate for PPISP main parameters."""

    ppisp_eps: float = 1e-15
    """Adam epsilon for PPISP main parameters."""

    ppisp_betas: tuple[float, float] = (0.9, 0.999)
    """Adam betas for PPISP main parameters."""

    # Optimizer settings for controller params
    controller_lr: float = 0.001
    """Learning rate for controller parameters (fixed, no scheduler)."""

    controller_eps: float = 1e-15
    """Adam epsilon for controller parameters."""

    controller_betas: tuple[float, float] = (0.9, 0.999)
    """Adam betas for controller parameters."""

    # Scheduler settings (PPISP main params only)
    scheduler_type: str = "linear_exp"
    """Scheduler type: 'linear_exp' for linear warmup + exponential decay."""

    scheduler_base_lr: float = 0.002
    """Base learning rate for scheduler (should match ppisp_lr)."""

    scheduler_warmup_steps: int = 500
    """Number of warmup steps for linear warmup phase."""

    scheduler_start_factor: float = 0.01
    """Starting factor for warmup (lr starts at start_factor * base_lr)."""

    scheduler_decay_max_steps: int = 30000
    """Number of steps for exponential decay to reach final_factor."""

    scheduler_final_factor: float = 0.01
    """Final factor for decay (lr decays toward final_factor * base_lr)."""


# Default configuration
DEFAULT_PPISP_CONFIG = PPISPConfig()


# =============================================================================
# Autograd Functions
# =============================================================================

class _PPISPFunction(torch.autograd.Function):
    """Custom autograd function for PPISP forward/backward."""

    @staticmethod
    def forward(
        ctx,
        exposure_params: torch.Tensor,
        vignetting_params: torch.Tensor,
        color_params: torch.Tensor,
        crf_params: torch.Tensor,
        rgb_in: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution_w: int,
        resolution_h: int,
        camera_idx: int,
        frame_idx: int,
    ) -> torch.Tensor:
        with torch.cuda.device(rgb_in.device):
            rgb_out = _C.ppisp_forward(
                exposure_params,
                vignetting_params,
                color_params,
                crf_params,
                rgb_in,
                pixel_coords,
                resolution_w,
                resolution_h,
                camera_idx,
                frame_idx,
            )

        ctx.save_for_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, rgb_out, pixel_coords
        )
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h
        ctx.camera_idx = camera_idx
        ctx.frame_idx = frame_idx

        return rgb_out

    @staticmethod
    def backward(ctx, v_rgb_out: torch.Tensor):
        (exposure_params, vignetting_params,
         color_params, crf_params, rgb_in, rgb_out, pixel_coords) = ctx.saved_tensors

        with torch.cuda.device(rgb_in.device):
            (v_exposure_params, v_vignetting_params,
             v_color_params, v_crf_params, v_rgb_in) = _C.ppisp_backward(
                exposure_params,
                vignetting_params,
                color_params,
                crf_params,
                rgb_in,
                rgb_out,
                pixel_coords,
                v_rgb_out.contiguous(),
                ctx.resolution_w,
                ctx.resolution_h,
                ctx.camera_idx,
                ctx.frame_idx,
            )

        return (
            v_exposure_params,
            v_vignetting_params,
            v_color_params,
            v_crf_params,
            v_rgb_in,
            None,  # pixel_coords
            None,  # resolution_w
            None,  # resolution_h
            None,  # camera_idx
            None,  # frame_idx
        )


# =============================================================================
# Public Functions
# =============================================================================

def ppisp_apply(
    exposure_params: torch.Tensor,
    vignetting_params: torch.Tensor,
    color_params: torch.Tensor,
    crf_params: torch.Tensor,
    rgb_in: torch.Tensor,
    pixel_coords: torch.Tensor,
    resolution_w: int,
    resolution_h: int,
    camera_idx: torch.Tensor | int | None = None,
    frame_idx: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Apply PPISP processing pipeline to input RGB.

    Tensor shapes are flattened on input and restored in the output. The last dimension
    of rgb_in must be 3 (RGB channels), and the last dimension of pixel_coords
    must be 2 (x, y).

    Args:
        exposure_params: Per-frame exposure [num_frames] or [1] from controller
        vignetting_params: Per-camera vignetting [num_cameras, 3, 5]
        color_params: Per-frame color correction [num_frames, 8]
        crf_params: Per-camera CRF [num_cameras, 3, 4]
        rgb_in: Input RGB [..., 3]
        pixel_coords: Pixel coordinates [..., 2]
        resolution_w: Image width
        resolution_h: Image height
        camera_idx: Camera index (Tensor, int, or None). None disables per-camera effects.
        frame_idx: Frame index (Tensor, int, or None). None disables per-frame effects.

    Returns:
        Processed RGB [..., 3] - same shape as rgb_in
    """
    # Normalize indices: Tensor/int/None -> int (-1 for None)
    camera_idx = _normalize_index(camera_idx, "camera_idx")
    frame_idx = _normalize_index(frame_idx, "frame_idx")

    # Store original shape for restoring output
    original_shape = rgb_in.shape

    # Flatten to [N, 3] and [N, 2] for processing
    rgb_flat = rgb_in.view(-1, rgb_in.shape[-1])
    coords_flat = pixel_coords.view(-1, pixel_coords.shape[-1])

    # Assertions on flattened tensors
    assert rgb_flat.shape[-1] == 3, f"Expected 3 RGB channels, got {rgb_flat.shape[-1]}"
    assert coords_flat.shape[-1] == 2, f"Expected 2D coords, got {coords_flat.shape[-1]}"
    assert rgb_flat.shape[0] == coords_flat.shape[0], (
        f"rgb and pixel_coords must have same num_pixels after flattening, "
        f"got {rgb_flat.shape[0]} vs {coords_flat.shape[0]}"
    )

    # Convert to float32 and ensure contiguous memory layout
    exposure_params = exposure_params.float().contiguous()
    vignetting_params = vignetting_params.float().contiguous()
    color_params = color_params.float().contiguous()
    crf_params = crf_params.float().contiguous()
    rgb_flat = rgb_flat.float().contiguous()
    coords_flat = coords_flat.float().contiguous()

    rgb_out = _PPISPFunction.apply(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_flat,
        coords_flat,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    )

    # Restore output to original shape
    return rgb_out.view(original_shape)


# =============================================================================
# Controller Network
# =============================================================================

class _PPISPController(nn.Module):
    """CNN-based controller that predicts exposure and color params from rendered radiance images.

    Fixed architecture:
    - CNN feature extraction with 1x1 convolutions
    - Adaptive average pooling to fixed spatial grid
    - MLP head for exposure + color parameter prediction
    - Uses prior exposure (from EXIF metadata) as additional input

    Input: Rendered radiance image + prior exposure
    Output: Exposure scalar + 8 color correction parameters
    """

    def __init__(
        self,
        input_downsampling: int = 3,
        cnn_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_mlp_layers: int = 3,
        pool_grid_size: tuple = (5, 5),
        device: str = "cuda",
    ):
        super().__init__()

        # CNN encoder: 1x1 convolutions for per-pixel feature extraction
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, device=device),
            nn.MaxPool2d(kernel_size=input_downsampling,
                         stride=input_downsampling),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=1, device=device),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cnn_feature_dim, kernel_size=1, device=device),
            nn.AdaptiveAvgPool2d(pool_grid_size),
            nn.Flatten(),
        )

        # Input dimension: CNN features + prior exposure
        cnn_output_dim = cnn_feature_dim * \
            pool_grid_size[0] * pool_grid_size[1]
        input_dim = cnn_output_dim + 1  # +1 for prior_exposure

        # MLP trunk (shared hidden layers)
        trunk_layers = [nn.Linear(input_dim, hidden_dim, device=device),
                        nn.ReLU(inplace=True)]
        for _ in range(num_mlp_layers - 1):
            trunk_layers.extend(
                [nn.Linear(hidden_dim, hidden_dim, device=device), nn.ReLU(inplace=True)])
        self.mlp_trunk = nn.Sequential(*trunk_layers)

        # Separate output heads (avoids misaligned views from slicing)
        self.exposure_head = nn.Linear(hidden_dim, 1, device=device)
        self.color_head = nn.Linear(
            hidden_dim, COLOR_PARAMS_PER_FRAME, device=device)

    def forward(
        self,
        rgb: torch.Tensor,
        prior_exposure: torch.Tensor | None = None,
    ) -> tuple:
        """Predict exposure and color parameters from rendered radiance image.

        Args:
            rgb: Rendered radiance image [H, W, 3]
            prior_exposure: Prior exposure from EXIF [1] (defaults to zero if not provided)

        Returns:
            exposure: Predicted exposure scalar
            color_params: Predicted color parameters [8]
        """
        # Default prior exposure to zero if not provided
        if prior_exposure is None:
            prior_exposure = torch.zeros(1, device=rgb.device)

        # Extract CNN features
        features = self.cnn_encoder(rgb.permute(
            2, 0, 1).unsqueeze(0).detach())  # [1, cnn_output_dim]
        features = torch.cat([features.squeeze(0), prior_exposure], dim=0)
        hidden = self.mlp_trunk(features)
        return self.exposure_head(hidden).squeeze(-1), self.color_head(hidden)

# =============================================================================
# Main PPISP Module
# =============================================================================


class PPISP(nn.Module):
    """Physically-Plausible Image Signal Processing module with integrated controller.

    Combines learned ISP parameters with a CNN-based controller for
    predicting exposure and color correction from rendered radiance images.

    The controller automatically activates at a specified fraction of training
    (controller_activation_ratio * max_optimization_iters). When activated,
    per-camera vignetting and CRF parameters are frozen.

    Note: Call create_schedulers() before forward() to set up the scheduler
    reference used for controller activation timing.

    Args:
        num_cameras: Number of cameras in the scene
        num_frames: Total number of frames across all cameras
        config: PPISP configuration (regularization, optimizer, scheduler settings)
    """

    def __init__(
        self,
        num_cameras: int,
        num_frames: int,
        config: PPISPConfig = DEFAULT_PPISP_CONFIG,
    ):
        super().__init__()

        self.config = config

        # Warn if controller is enabled but will never train (ratio >= 1.0)
        if config.use_controller and config.controller_activation_ratio >= 1.0:
            print(
                f"[PPISP] Warning: controller_activation_ratio="
                f"{config.controller_activation_ratio} >= 1.0. "
                "Controller will not be trained. "
                "Zero per-frame corrections will be used for novel views (frame_idx=-1)."
            )

        # Scheduler reference (set by create_schedulers, not serialized)
        self._ppisp_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None

        # Controller activation step (set by create_schedulers)
        # -1 means not yet set
        self._controller_activation_step: int = -1

        # Per-frame exposure offset (in log space)
        self.exposure_params = nn.Parameter(
            torch.zeros(num_frames, device="cuda")
        )

        # Per-camera vignetting: [center_x, center_y, alpha_0, alpha_1, alpha_2] per channel
        # Per-channel vignetting [num_cameras, 3, 5]
        self.vignetting_params = nn.Parameter(
            torch.zeros(num_cameras, 3,
                        VIGNETTING_PARAMS_PER_CHANNEL, device="cuda")
        )

        # Per-frame color params [num_frames, 8]
        # [db_r, db_g, dr_r, dr_g, dg_r, dg_g, dgray_r, dgray_g]
        self.color_params = nn.Parameter(
            torch.zeros(num_frames, COLOR_PARAMS_PER_FRAME, device="cuda")
        )

        # Per-camera CRF: [toe_raw, shoulder_raw, gamma_raw, center_raw] per channel
        # Initialize to identity-like response
        def softplus_inverse(x: float, min_value: float = 0.0, epsilon: float = 1e-5) -> float:
            clamped_value = max(epsilon, x - min_value)
            return float(torch.log(torch.expm1(torch.tensor(clamped_value))))

        crf_raw = torch.zeros(CRF_PARAMS_PER_CHANNEL, device="cuda")
        crf_raw[0] = softplus_inverse(1.0, min_value=0.3)  # toe
        crf_raw[1] = softplus_inverse(1.0, min_value=0.3)  # shoulder
        crf_raw[2] = softplus_inverse(1.0, min_value=0.1)  # gamma
        crf_raw[3] = 0.0  # center_raw -> sigmoid(0) = 0.5

        self.crf_params = nn.Parameter(
            crf_raw.view(1, 1, CRF_PARAMS_PER_CHANNEL)
            .repeat(num_cameras, 3, 1)
            .contiguous()
        )

        # ZCA pinv block-diagonal matrix for color loss computation (constant, non-persistent)
        # 8x8 block-diagonal: each 2x2 block transforms latent (dr, dg) to real chromaticity offsets
        # Order: [Blue, Red, Green, Neutral]
        self.register_buffer(
            "color_pinv_block_diag",
            _COLOR_PINV_BLOCK_DIAG.to(device="cuda"),
            persistent=False,
        )

        # Controllers for predicting per-frame corrections
        if config.use_controller:
            self.controllers = nn.ModuleList([
                _PPISPController() for _ in range(num_cameras)
            ])
        else:
            self.controllers = nn.ModuleList()

    @property
    def num_cameras(self) -> int:
        """Number of cameras, inferred from crf_params shape."""
        return self.crf_params.shape[0]

    @property
    def num_frames(self) -> int:
        """Number of frames, inferred from exposure_params shape."""
        return self.exposure_params.shape[0]

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        config: PPISPConfig = DEFAULT_PPISP_CONFIG,
    ) -> "PPISP":
        """Create PPISP instance from a state dict.

        Infers num_cameras and num_frames from parameter shapes in the state dict.
        This enables checkpoint restoration without needing to store dimensions explicitly.

        Args:
            state_dict: State dict from a saved PPISP module (via state_dict())
            config: PPISP configuration (default: DEFAULT_PPISP_CONFIG)

        Returns:
            New PPISP instance with loaded state
        """
        # Infer dimensions from parameter shapes
        # Use crf_params for num_cameras
        num_cameras = state_dict["crf_params"].shape[0]
        num_frames = state_dict["exposure_params"].shape[0]

        # Create instance with inferred dimensions
        instance = cls(num_cameras=num_cameras,
                       num_frames=num_frames, config=config)
        instance.load_state_dict(state_dict)

        return instance

    def forward(
        self,
        rgb: torch.Tensor,
        pixel_coords: torch.Tensor,
        resolution: tuple[int, int],
        camera_idx: torch.Tensor | int | None = None,
        frame_idx: torch.Tensor | int | None = None,
        exposure_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply PPISP processing to input RGB.

        Tensor shapes are flattened on input and restored in the output.
        The last dimension of rgb must be 3 (RGB channels), and the last
        dimension of pixel_coords must be 2 (x, y).

        Args:
            rgb: Input RGB [..., 3]
            pixel_coords: Pixel coordinates [..., 2]
            resolution: Image resolution as (width, height)
            camera_idx: Camera index (Tensor, int, or None). None disables per-camera effects.
            frame_idx: Frame index (Tensor, int, or None). None disables per-frame effects.
            exposure_prior: Prior exposure value [1] (defaults to zero if not provided)

        Returns:
            Processed RGB [..., 3] - same shape as rgb
        """
        resolution_w, resolution_h = resolution

        # Normalize indices: Tensor/int/None -> int (-1 for None)
        camera_idx_int = _normalize_index(camera_idx, "camera_idx")
        frame_idx_int = _normalize_index(frame_idx, "frame_idx")

        # Validate indices against parameter dimensions
        if camera_idx_int != -1:
            assert 0 <= camera_idx_int < self.num_cameras, f"Invalid camera_idx: {camera_idx_int}"
        if frame_idx_int != -1:
            assert 0 <= frame_idx_int < self.num_frames, f"Invalid frame_idx: {frame_idx_int}"

        # Check if controller is trained (enabled AND step threshold reached)
        # Controller is not trained if: disabled, ratio >= 1, or training hasn't reached
        # activation step yet. During inference (no scheduler), assume controller is ready.
        if self._ppisp_scheduler is None:
            # Inference mode: controller is ready if enabled and exists
            controller_trained = (
                self.config.use_controller
                and self.config.controller_activation_ratio < 1.0
                and len(self.controllers) > 0
            )
        else:
            # Training mode: check activation step threshold
            controller_trained = (
                self.config.use_controller
                and self.config.controller_activation_ratio < 1.0
                and self._controller_activation_step >= 0
                and self._ppisp_scheduler.last_epoch >= self._controller_activation_step
            )

        if controller_trained and self.config.controller_distillation:
            # Distillation: freeze the other PPISP parameters
            self.exposure_params.requires_grad = False
            self.vignetting_params.requires_grad = False
            self.color_params.requires_grad = False
            self.crf_params.requires_grad = False
            # Distillation: detach the input
            rgb = rgb.detach()

        # Determine if we should apply correction overrides
        # Overrides are used for:
        # 1. Novel views (frame_idx=-1): either controller predictions or zeros
        # 2. Training after controller activation: controller predictions
        is_novel_view = (camera_idx_int != -1 and frame_idx_int == -1)
        apply_correction_override = is_novel_view or (
            controller_trained and camera_idx_int != -1)

        if apply_correction_override:
            if is_novel_view and not controller_trained:
                # Novel view with untrained controller: apply identity corrections
                exposure = torch.zeros(1, device=rgb.device, dtype=rgb.dtype)
                # Homography offsets, identity is zeros
                color = torch.zeros(
                    1, COLOR_PARAMS_PER_FRAME, device=rgb.device, dtype=rgb.dtype
                )
            else:
                # Use controller predictions
                # View rgb as [H, W, 3] image for controller
                rgb_image = rgb.view(resolution_h, resolution_w, 3)

                # Predict exposure and color params using controller
                controller = self.controllers[camera_idx_int]
                exposure_pred, color_pred = controller(
                    rgb_image, exposure_prior)

                # Use 1-element tensors for predictions to save memory bandwidth
                exposure = exposure_pred.unsqueeze(0)  # [1]
                color = color_pred.unsqueeze(0)  # [1, 8]

            # When override is applied, kernel always reads from index 0
            frame_idx_for_kernel = 0
        else:
            exposure = self.exposure_params
            color = self.color_params
            frame_idx_for_kernel = frame_idx_int

        return ppisp_apply(
            exposure,
            self.vignetting_params,
            color,
            self.crf_params,
            rgb,
            pixel_coords,
            resolution_w,
            resolution_h,
            camera_idx_int,
            frame_idx_for_kernel,
        )

    def get_regularization_loss(self) -> torch.Tensor:
        """Compute weighted regularization loss for PPISP parameters.

        Regularization terms:
            - Exposure mean regularization
            - Vignetting center/channel/non-positivity regularization
            - Color mean regularization
            - CRF channel variance regularization

        Returns:
            Single scalar tensor with the total weighted regularization loss.
        """
        cfg = self.config
        total_loss = torch.tensor(0.0, device=self.exposure_params.device)

        # Exposure mean regularization (fix SH <-> exposure ambiguity)
        if cfg.exposure_mean > 0:
            exposure_residual = self.exposure_params.mean()
            total_loss = total_loss + cfg.exposure_mean * F.smooth_l1_loss(
                exposure_residual, torch.zeros_like(exposure_residual), beta=0.1
            )

        # Vignetting center loss: optical center should be near image center (0, 0)
        if cfg.vig_center > 0:
            # [num_cameras, 3, 2]
            vig_optical_center = self.vignetting_params[:, :, :2]
            # [num_cameras, 3]
            vig_center = (vig_optical_center ** 2).sum(dim=-1)
            total_loss = total_loss + cfg.vig_center * vig_center.mean()

        # Vignetting non-positivity loss: alpha coefficients should be <= 0
        if cfg.vig_non_pos > 0:
            # [num_cameras, 3, NUM_VIGNETTING_ALPHA_TERMS]
            vig_alphas = self.vignetting_params[:, :, 2:]
            vig_non_pos = F.relu(vig_alphas)  # penalty for positive values
            total_loss = total_loss + cfg.vig_non_pos * vig_non_pos.mean()

        # Vignetting channel variance
        if cfg.vig_channel > 0:
            total_loss = total_loss + cfg.vig_channel * \
                self.vignetting_params.var(dim=1, unbiased=False).mean()

        # Color mean regularization using ZCA block-diagonal matrix
        if cfg.color_mean > 0:
            color_offsets = self.color_params @ self.color_pinv_block_diag
            color_residual = color_offsets.mean(dim=0)
            total_loss = total_loss + cfg.color_mean * F.smooth_l1_loss(
                color_residual, torch.zeros_like(color_residual), beta=0.005, reduction="mean"
            )

        # CRF channel variance
        if cfg.crf_channel > 0:
            total_loss = total_loss + cfg.crf_channel * \
                self.crf_params.var(dim=1, unbiased=False).mean()

        return total_loss

    def create_optimizers(self) -> list[torch.optim.Optimizer]:
        """Create optimizers for PPISP parameters.

        Returns Adam optimizers:
        1. PPISP main params optimizer (exposure, vignetting, color, CRF)
        2. Controller params optimizer (neural network weights) - only if use_controller=True

        Returns:
            List of optimizers: [ppisp_optimizer] or [ppisp_optimizer, controller_optimizer]
        """
        cfg = self.config

        # PPISP main parameters
        ppisp_params = [
            self.exposure_params,
            self.vignetting_params,
            self.color_params,
            self.crf_params,
        ]

        ppisp_optimizer = torch.optim.Adam(
            ppisp_params,
            lr=cfg.ppisp_lr,
            eps=cfg.ppisp_eps,
            betas=cfg.ppisp_betas,
        )

        optimizers = [ppisp_optimizer]

        # Controller parameters (only if controller is enabled)
        if cfg.use_controller and len(self.controllers) > 0:
            controller_optimizer = torch.optim.Adam(
                self.controllers.parameters(),
                lr=cfg.controller_lr,
                eps=cfg.controller_eps,
                betas=cfg.controller_betas,
            )
            optimizers.append(controller_optimizer)

        return optimizers

    def create_schedulers(
        self,
        optimizers: list[torch.optim.Optimizer],
        max_optimization_iters: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        """Create learning rate schedulers for the optimizers.

        Only the PPISP main params optimizer gets a scheduler (linear warmup + exp decay).
        The controller optimizer uses a fixed learning rate and has no scheduler.

        This method also stores the scheduler reference internally so that forward()
        can read the current step for controller activation timing.

        Args:
            optimizers: List of optimizers from create_optimizers()
                        [ppisp_optimizer] or [ppisp_optimizer, controller_optimizer]
            max_optimization_iters: Total number of optimization steps for training.
                        Used to compute controller activation step.

        Returns:
            List containing the PPISP scheduler.
        """
        cfg = self.config
        ppisp_optimizer = optimizers[0]

        # Compute controller activation step
        self._controller_activation_step = int(
            cfg.controller_activation_ratio * max_optimization_iters)

        # Compute decay gamma for exponential decay phase
        # After decay_max_steps, lr should be final_factor * base_lr
        gamma = cfg.scheduler_final_factor ** (1.0 /
                                               cfg.scheduler_decay_max_steps)

        # Linear warmup + exponential decay using built-in schedulers
        ppisp_scheduler = torch.optim.lr_scheduler.SequentialLR(
            ppisp_optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    ppisp_optimizer,
                    start_factor=cfg.scheduler_start_factor,
                    total_iters=cfg.scheduler_warmup_steps,
                ),
                torch.optim.lr_scheduler.ExponentialLR(
                    ppisp_optimizer,
                    gamma=gamma,
                ),
            ],
            milestones=[cfg.scheduler_warmup_steps],
        )

        # Store reference so forward() can read current step
        self._ppisp_scheduler = ppisp_scheduler

        return [ppisp_scheduler]

```



### File: ppisp\bindings.h

```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _PPISP_BINDINGS_H_INC
#define _PPISP_BINDINGS_H_INC

#include <torch/extension.h>

// =============================================================================
// Forward pass for PPISP image processing
// =============================================================================

void ppisp_forward(
    // Parameters (per-camera/per-frame)
    const float *exposure_params,    // [num_frames]
    const float *vignetting_params,  // [num_cameras, 3, 5]
    const float *color_params,       // [num_frames, 8]
    const float *crf_params,         // [num_cameras, 3, 4]
    // Input/Output
    const float *rgb_in,        // [num_pixels, 3]
    float *rgb_out,             // [num_pixels, 3]
    const float *pixel_coords,  // [num_pixels, 2]
    // Dimensions
    int num_pixels, int num_cameras, int num_frames, int resolution_w, int resolution_h,
    int camera_idx, int frame_idx);

// =============================================================================
// Backward pass for PPISP image processing
// =============================================================================

void ppisp_backward(
    // Parameters (per-camera/per-frame)
    const float *exposure_params, const float *vignetting_params, const float *color_params,
    const float *crf_params,
    // Input/Output from forward
    const float *rgb_in, const float *rgb_out, const float *pixel_coords,
    // Gradient of loss w.r.t. output
    const float *v_rgb_out,
    // Gradients w.r.t. parameters
    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
    float *v_crf_params, float *v_rgb_in,
    // Dimensions
    int num_pixels, int num_cameras, int num_frames, int resolution_w, int resolution_h,
    int camera_idx, int frame_idx);

// =============================================================================
// PyTorch tensor wrappers
// =============================================================================

torch::Tensor ppisp_forward_tensor(torch::Tensor exposure_params,    // [num_frames]
                                   torch::Tensor vignetting_params,  // [num_cameras, 3, 5]
                                   torch::Tensor color_params,       // [num_frames, 8]
                                   torch::Tensor crf_params,         // [num_cameras, 3, 4]
                                   torch::Tensor rgb_in,             // [num_pixels, 3]
                                   torch::Tensor pixel_coords,       // [num_pixels, 2]
                                   int resolution_w, int resolution_h, int camera_idx,
                                   int frame_idx) {
    int num_pixels = rgb_in.size(0);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto rgb_out = torch::empty_like(rgb_in);

    ppisp_forward(exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
                  color_params.data_ptr<float>(), crf_params.data_ptr<float>(),
                  rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(),
                  pixel_coords.data_ptr<float>(), num_pixels, num_cameras, num_frames, resolution_w,
                  resolution_h, camera_idx, frame_idx);

    return rgb_out;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
ppisp_backward_tensor(torch::Tensor exposure_params, torch::Tensor vignetting_params,
                      torch::Tensor color_params, torch::Tensor crf_params, torch::Tensor rgb_in,
                      torch::Tensor rgb_out, torch::Tensor pixel_coords, torch::Tensor v_rgb_out,
                      int resolution_w, int resolution_h, int camera_idx, int frame_idx) {
    int num_pixels = rgb_in.size(0);
    int num_cameras = crf_params.size(0);
    int num_frames = exposure_params.size(0);

    auto v_exposure_params = torch::zeros_like(exposure_params);
    auto v_vignetting_params = torch::zeros_like(vignetting_params);
    auto v_color_params = torch::zeros_like(color_params);
    auto v_crf_params = torch::zeros_like(crf_params);
    auto v_rgb_in = torch::zeros_like(rgb_in);

    ppisp_backward(exposure_params.data_ptr<float>(), vignetting_params.data_ptr<float>(),
                   color_params.data_ptr<float>(), crf_params.data_ptr<float>(),
                   rgb_in.data_ptr<float>(), rgb_out.data_ptr<float>(),
                   pixel_coords.data_ptr<float>(), v_rgb_out.data_ptr<float>(),
                   v_exposure_params.data_ptr<float>(), v_vignetting_params.data_ptr<float>(),
                   v_color_params.data_ptr<float>(), v_crf_params.data_ptr<float>(),
                   v_rgb_in.data_ptr<float>(), num_pixels, num_cameras, num_frames, resolution_w,
                   resolution_h, camera_idx, frame_idx);

    return std::make_tuple(v_exposure_params, v_vignetting_params, v_color_params, v_crf_params,
                           v_rgb_in);
}

#endif  // _PPISP_BINDINGS_H_INC

```



### File: ppisp\ext.cpp

```cpp
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ppisp_forward", &ppisp_forward_tensor);
    m.def("ppisp_backward", &ppisp_backward_tensor);
}

```



### File: ppisp\report.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PPISP Report Export

Export PDF reports visualizing learned PPISP parameters:
- Exposure compensation (per-frame)
- Vignetting (per-camera, per-channel)
- Color correction (per-frame)
- Camera Response Function / tone mapping (per-camera)
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from . import __version__

if TYPE_CHECKING:
    from . import PPISP


def _srgb_inverse_oetf(x: np.ndarray) -> np.ndarray:
    """Inverse sRGB OETF (EOTF): pre-compensation so that a later sRGB OETF yields identity."""
    x = np.clip(x.astype(np.float32), 0.0, 1.0)
    out = np.empty_like(x, dtype=np.float32)
    mask = x <= 0.04045
    out[mask] = x[mask] / 12.92
    inv = x[~mask]
    out[~mask] = np.power((inv + 0.055) / 1.055, 2.4, dtype=np.float32)
    return out


def _gray_bars(size: int = 256, num: int = 16) -> np.ndarray:
    w = size // num
    img = np.zeros((size, size, 3), dtype=np.float32)
    for i in range(num):
        v = i / (num - 1)
        img[:, i * w: size if i == num - 1 else (i + 1) * w] = v
    return img


def _show_image(ax, img: np.ndarray, title: str):
    ax.imshow(np.clip(img, 0.0, 1.0))
    ax.axis("off")
    ax.set_title(title)


# =============================================================================
# Exposure Plotting
# =============================================================================


def _plot_exposure(
    fig,
    gs,
    exposure_params: torch.Tensor,
    frames_per_camera: list[int],
    cam: int,
):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    start = int(sum(frames_per_camera[:cam]))
    end = start + int(frames_per_camera[cam])
    vals = exposure_params[start:end].detach().float().cpu()
    mean_val = vals.mean().item() if vals.numel() else 0.0

    ax_plot.plot(np.arange(vals.numel()), vals.numpy(), "b-")
    ax_plot.axhline(mean_val, color="b", linestyle="--", alpha=0.5)
    ax_plot.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Frame Index")
    ax_plot.set_ylabel("Exposure Offset [EV]")
    ax_plot.set_title("Exposure Offset Over Time")
    ax_plot.grid(True, alpha=0.3)

    img = _gray_bars(256)
    scale = 2.0 ** float(mean_val)
    img[img.shape[0] // 2:, :] *= scale
    _show_image(ax_img, img, "Mean Exposure Visualization")
    size = img.shape[0]
    ax_img.text(
        size * 0.5, 20, "Original", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, f"{mean_val:+.2f} EV", ha="center", va="bottom",
        color="white", bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# Vignetting Plotting
# =============================================================================


def _vig_weight_forward(r2: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
    """Exact forward falloff: 1 + sum a_i * r2^(i+1), clamped to [0,1]."""
    falloff = torch.ones_like(r2)
    r2_pow = r2
    for i in range(int(alphas.shape[-1])):
        falloff = falloff + alphas[..., i] * r2_pow
        r2_pow = r2_pow * r2
    return torch.clamp(falloff, 0.0, 1.0)


def _plot_vignetting(fig, gs, vig_params: torch.Tensor, cam: int):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    device = vig_params.device
    r = torch.linspace(0, np.sqrt(2) / 2.0, 200, device=device)
    r2 = r * r

    colors = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    # Per-channel vignetting curves
    for ch in range(3):
        alphas = vig_params[cam, ch, 2:]
        w = _vig_weight_forward(r2, alphas)
        ax_plot.plot(
            r.detach().cpu().numpy(), w.detach().cpu().numpy(),
            color=colors[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch],
        )
    ax_plot.set_title("Vignetting Curves (R,G,B)")

    ax_plot.set_xlabel("Radial Distance")
    ax_plot.set_ylabel("Light Transmission")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()
    ax_plot.set_ylim(bottom=0)

    # Vignette visualization on square uv grid centered at 0
    size = 256
    coords = torch.linspace(-0.5, 0.5, size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    img = torch.full((size, size, 3), 0.75, device=device)

    # Per-channel vignetting
    for ch in range(3):
        oc = vig_params[cam, ch, :2]
        alphas = vig_params[cam, ch, 2:]
        dx = xx - oc[0]
        dy = yy - oc[1]
        r2m = dx * dx + dy * dy
        w = _vig_weight_forward(r2m, alphas)
        img[..., ch] = img[..., ch] * w

    img_np = img.detach().cpu().numpy()
    ax_img.imshow(np.clip(img_np, 0.0, 1.0))
    ax_img.axis("off")
    ax_img.set_title("Vignetting Effect Visualization")
    center = size * 0.5
    ax_img.axhline(y=center, color="gray", linestyle="--", alpha=0.5)
    ax_img.axvline(x=center, color="gray", linestyle="--", alpha=0.5)
    # Overlay crosses at optical centers
    cross_size = 10
    cross_width = 2
    for ch, color in enumerate(colors):
        oc = vig_params[cam, ch, :2].detach().cpu().numpy()
        cx = (float(oc[0]) + 0.5) * size
        cy = (float(oc[1]) + 0.5) * size
        ax_img.plot([cx - cross_size, cx + cross_size],
                    [cy, cy], color=color, linewidth=cross_width)
        ax_img.plot([cx, cx], [cy - cross_size, cy + cross_size],
                    color=color, linewidth=cross_width)


# =============================================================================
# Color Correction Plotting (Standard PPISP)
# =============================================================================

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
_PINV_BLOCKS = torch.tensor([
    [[0.0480542, -0.0043631], [-0.0043631, 0.0481283]],
    [[0.0580570, -0.0179872], [-0.0179872, 0.0431061]],
    [[0.0433336, -0.0180537], [-0.0180537, 0.0580500]],
    [[0.0128369, -0.0034654], [-0.0034654, 0.0128158]],
])


def _color_offsets_from_params(p: torch.Tensor) -> torch.Tensor:
    """Map latent 8-dim color params per frame to real RG offsets via ZCA blocks.

    Args:
        p: [num_frames, 8] tensor ordered as pairs per chromaticity:
           [blue(dr,dg), red(dr,dg), green(dr,dg), neutral(dr,dg)]

    Returns:
        Tensor of shape [num_frames, 4, 2] with real-space RG offsets for
        [blue, red, green, neutral] in that order.
    """
    device = p.device
    dtype = p.dtype
    pinv = _PINV_BLOCKS.to(device=device, dtype=dtype)

    def _mul2(a: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return torch.stack([
            a[..., 0] * m[0, 0] + a[..., 1] * m[0, 1],
            a[..., 0] * m[1, 0] + a[..., 1] * m[1, 1],
        ], dim=-1)

    xb, xr, xg, xn = p[..., 0:2], p[..., 2:4], p[..., 4:6], p[..., 6:8]
    yb = _mul2(xb, pinv[0])
    yr = _mul2(xr, pinv[1])
    yg = _mul2(xg, pinv[2])
    yn = _mul2(xn, pinv[3])
    return torch.stack([yb, yr, yg, yn], dim=1)


def _source_chroms(device: torch.device) -> torch.Tensor:
    # RG chromaticities for [Blue, Red, Green, Neutral]
    return torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.33, 0.33]], device=device)


def _homography_from_params(p: torch.Tensor) -> torch.Tensor:
    """Construct H from RG offsets p for [blue, red, green, gray].

    p shape: [..., 8] ordered as pairs per chromaticity.
    """
    device = p.device
    dtype = p.dtype

    offsets = _color_offsets_from_params(p)  # [...,4,2]
    bd = offsets[..., 0, :]
    rd = offsets[..., 1, :]
    gd = offsets[..., 2, :]
    nd = offsets[..., 3, :]

    # Sources
    s_b = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=device, dtype=dtype)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=device, dtype=dtype)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0],
                          device=device, dtype=dtype)

    t_b = torch.stack([s_b[0] + bd[..., 0], s_b[1] +
                      bd[..., 1], torch.ones_like(bd[..., 0])], dim=-1)
    t_r = torch.stack([s_r[0] + rd[..., 0], s_r[1] +
                      rd[..., 1], torch.ones_like(rd[..., 0])], dim=-1)
    t_g = torch.stack([s_g[0] + gd[..., 0], s_g[1] +
                      gd[..., 1], torch.ones_like(gd[..., 0])], dim=-1)
    t_gray = torch.stack([s_gray[0] + nd[..., 0], s_gray[1] +
                         nd[..., 1], torch.ones_like(nd[..., 0])], dim=-1)

    T = torch.stack([t_b, t_r, t_g], dim=-1)  # [...,3,3]

    zero = torch.zeros_like(bd[..., 0])
    skew = torch.stack([
        torch.stack([zero, -t_gray[..., 2], t_gray[..., 1]], dim=-1),
        torch.stack([t_gray[..., 2], zero, -t_gray[..., 0]], dim=-1),
        torch.stack([-t_gray[..., 1], t_gray[..., 0], zero], dim=-1),
    ], dim=-2)

    M = torch.matmul(skew, T)
    r0, r1, r2 = M[..., 0, :], M[..., 1, :], M[..., 2, :]
    lam = torch.cross(r0, r1, dim=-1)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r0, r2, dim=-1), lam)
    n2 = (lam * lam).sum(dim=-1)
    mask = n2 < 1.0e-20
    lam = torch.where(mask.unsqueeze(-1), torch.cross(r1, r2, dim=-1), lam)

    S_inv = torch.tensor([[-1.0, -1.0, 1.0], [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], device=device, dtype=dtype)
    D = torch.zeros(*p.shape[:-1], 3, 3, device=device, dtype=dtype)
    D[..., 0, 0] = lam[..., 0]
    D[..., 1, 1] = lam[..., 1]
    D[..., 2, 2] = lam[..., 2]
    H = torch.matmul(T, torch.matmul(D, S_inv))
    s = H[..., 2:3, 2:3]
    denom = s + (s.abs() <= 1.0e-20).to(dtype)
    H = H / denom
    return H


def _apply_h_rg_loss(h: torch.Tensor, rg: torch.Tensor) -> torch.Tensor:
    """Loss mapping: apply H to (r,g,1) and divide xy by z."""
    r, g = rg[..., 0], rg[..., 1]
    ones = torch.ones_like(r)
    v = torch.stack([r, g, ones], dim=-1)
    vv = torch.matmul(h, v.unsqueeze(-1)).squeeze(-1)
    denom = vv[..., 2] + 1.0e-5
    return torch.stack([vv[..., 0] / denom, vv[..., 1] / denom], dim=-1)


def _apply_h_rgb_forward(h: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Forward mapping: RGB -> RGI, apply H, intensity-preserving renormalization, back to RGB."""
    intensity = rgb[..., 0] + rgb[..., 1] + rgb[..., 2]
    rgi = torch.stack([rgb[..., 0], rgb[..., 1], intensity], dim=-1)
    rgi_m = torch.matmul(h, rgi.unsqueeze(-1)).squeeze(-1)
    scale = intensity / (rgi_m[..., 2] + 1.0e-5)
    rgi_m = rgi_m * scale.unsqueeze(-1)
    r, g = rgi_m[..., 0], rgi_m[..., 1]
    b = rgi_m[..., 2] - r - g
    return torch.stack([r, g, b], dim=-1)


def _dlt_homography(src_rg: np.ndarray, dst_rg: np.ndarray) -> np.ndarray:
    """Solve for H (3x3) mapping (r,g,1)->(r',g',1) using 4 RG correspondences via DLT."""
    A = []
    for (x, y), (u, v) in zip(src_rg, dst_rg):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A, dtype=np.float64)
    _, _, vh = np.linalg.svd(A)
    h = vh[-1]
    H = h.reshape(3, 3)
    if abs(H[2, 2]) < 1e-8:
        H = H / (np.sign(H[2, 2]) + 1e-8)
    else:
        H = H / H[2, 2]
    return H


def _chrom_triangle_size(size: int) -> tuple[int, int]:
    height = int(size * np.sqrt(3.0) / 2.0)
    return size, height


def _chrom_barycentric_to_window(r: float, g: float, size: int) -> tuple[float, float]:
    width, height = _chrom_triangle_size(size)
    top = (width * 0.5, 0.0)
    bl = (0.0, float(height))
    br = (float(width), float(height))
    b = 1.0 - r - g
    x = r * bl[0] + g * br[0] + b * top[0]
    y = r * bl[1] + g * br[1] + b * top[1]
    return x, y


@lru_cache(maxsize=4)
def _create_chromaticity_triangle(size: int) -> np.ndarray:
    width, height = _chrom_triangle_size(size)
    img = np.ones((height, width, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            top = np.array([width * 0.5, 0.0])
            bl = np.array([0.0, height])
            br = np.array([width, height])
            v0 = bl - top
            v1 = br - top
            v2 = np.array([x, y], dtype=np.float32) - top
            d00 = (v0 * v0).sum()
            d01 = (v0 * v1).sum()
            d11 = (v1 * v1).sum()
            d20 = (v2 * v0).sum()
            d21 = (v2 * v1).sum()
            denom = d00 * d11 - d01 * d01
            if denom == 0:
                continue
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            if (u >= 0.0) and (v >= 0.0) and (w >= 0.0):
                r_val, g_val = v, w
                b_val = max(0.0, 1.0 - r_val - g_val)
                rgb = np.array([r_val, g_val, b_val], dtype=np.float32)
                m = rgb.max()
                if m > 0:
                    rgb = rgb / m
                img[y, x] = rgb * 0.85 + 0.15
    return img


def _plot_color(
    fig,
    gs_top,
    gs_bot,
    color_params: torch.Tensor,
    frames_per_camera: list[int],
    cam: int,
):
    start = int(sum(frames_per_camera[:cam]))
    n = int(frames_per_camera[cam])
    p = color_params[start: start + n]
    H = _homography_from_params(p)
    src = _source_chroms(color_params.device)
    tgt_list = []
    for i in range(4):
        rg_in = src[i].unsqueeze(0).expand(n, -1)
        tgt_i = _apply_h_rg_loss(H, rg_in)
        tgt_list.append(tgt_i)
    tgt = torch.stack(tgt_list, dim=1)
    shifts = (tgt - src)

    names = ["Blue", "Red", "Green", "Neutral"]
    cols = ["blue", "red", "green", "gray"]

    sub_top = gs_top.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    sub_bot = gs_bot.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_rc = fig.add_subplot(sub_top[0])
    ax_rgplot = fig.add_subplot(sub_top[1])
    ax_gm = fig.add_subplot(sub_bot[0])
    ax_img = fig.add_subplot(sub_bot[1])

    x = np.arange(n)
    for i in range(4):
        ax_rc.plot(x, shifts[:, i, 0].detach().cpu().numpy(),
                   color=cols[i], label=names[i], alpha=0.8)
        ax_gm.plot(x, shifts[:, i, 1].detach(
        ).cpu().numpy(), color=cols[i], alpha=0.8)
    ax_rc.set_title("Red-Cyan Shift Over Time")
    ax_rc.set_xlabel("Frame Index")
    ax_rc.set_ylabel("Red-Cyan Shift")
    ax_rc.grid(True, alpha=0.3)
    ax_rc.legend()

    ax_gm.set_title("Green-Magenta Shift Over Time")
    ax_gm.set_xlabel("Frame Index")
    ax_gm.set_ylabel("Green-Magenta Shift")
    ax_gm.grid(True, alpha=0.3)

    size = 256
    scale = 5.0
    tri_img = _create_chromaticity_triangle(size)
    ax_rgplot.imshow(tri_img)
    ax_rgplot.axis("off")
    ax_rgplot.set_title(f"Chromaticity Shifts Over Time, Scaled {scale:.1f}x")

    chroms_scaled = (src + shifts * scale).detach().cpu().numpy()
    cross_size = 7
    cross_width = 2
    for i in range(4):
        pts = chroms_scaled[:, i, :]
        traj = np.array([_chrom_barycentric_to_window(
            float(r_val), float(g_val), size) for r_val, g_val in pts])
        ax_rgplot.plot(traj[:, 0], traj[:, 1], "-",
                       color="black", linewidth=1.0, alpha=0.7)
        fx, fy = traj[-1]
        ax_rgplot.plot([fx - cross_size, fx + cross_size],
                       [fy, fy], "-", color="black", linewidth=cross_width)
        ax_rgplot.plot([fx, fx], [fy - cross_size, fy + cross_size],
                       "-", color="black", linewidth=cross_width)
        sx, sy = _chrom_barycentric_to_window(
            float(src[i, 0].item()), float(src[i, 1].item()), size)
        ax_rgplot.plot(
            [sx - cross_size * 0.75, sx + cross_size * 0.75], [sy, sy],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5,
        )
        ax_rgplot.plot(
            [sx, sx], [sy - cross_size * 0.75, sy + cross_size * 0.75],
            "-", color="black", linewidth=cross_width / 2, alpha=0.5,
        )

    mean_targets = tgt.mean(dim=0).detach().cpu().numpy()
    src_np = src.detach().cpu().numpy()
    H_np = _dlt_homography(src_np, mean_targets)
    H_mean = torch.from_numpy(H_np).to(
        color_params.device, dtype=color_params.dtype)

    size = 256
    bars = np.zeros((size, size, 3), dtype=np.float32)
    w = size // 4
    bars[:, 0:w] = [0, 0, 1]
    bars[:, w: 2 * w] = [1, 0, 0]
    bars[:, 2 * w: 3 * w] = [0, 1, 0]
    bars[:, 3 * w:] = [0.5, 0.5, 0.5]
    bottom = torch.from_numpy(
        bars[size // 2:].reshape(-1, 3)).to(color_params.device)
    corrected = _apply_h_rgb_forward(H_mean, bottom)
    vis = bars.copy()
    vis[size // 2:] = corrected.reshape(size // 2,
                                        size, 3).clamp(0, 1).detach().cpu().numpy()
    _show_image(ax_img, vis, "Mean Color Correction Visualization")
    ax_img.text(
        size * 0.5, 20, "Original", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, f"Color Corrected, Scaled {scale:.1f}x", ha="center", va="bottom",
        color="white", bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# CRF Plotting
# =============================================================================


def _softplus_with_min(x: torch.Tensor, min_value: float) -> torch.Tensor:
    return torch.tensor(min_value, device=x.device, dtype=x.dtype) + torch.log1p(torch.exp(x))


def _crf_effective_from_raw(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map raw CRF params to effective toe, shoulder, gamma, center."""
    toe = _softplus_with_min(raw[..., 0], 0.3)
    shoulder = _softplus_with_min(raw[..., 1], 0.3)
    gamma = _softplus_with_min(raw[..., 2], 0.1)
    center = torch.sigmoid(raw[..., 3])
    return toe, shoulder, gamma, center


def _apply_crf(
    toe: torch.Tensor,
    shoulder: torch.Tensor,
    gamma: torch.Tensor,
    center: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Apply CRF with center split."""
    x = torch.clamp(x, 0.0, 1.0)
    a = (shoulder * center) / \
        torch.clamp(torch.lerp(toe, shoulder, center), min=1.0e-12)
    b = 1.0 - a
    left = a * \
        torch.pow(torch.clamp(
            x / torch.clamp(center, min=1.0e-12), 0.0, 1.0), toe)
    right = 1.0 - b * torch.pow(torch.clamp((1.0 - x) /
                                torch.clamp(1.0 - center, min=1.0e-12), 0.0, 1.0), shoulder)
    y0 = torch.where(x <= center, left, right)
    y = torch.pow(torch.clamp(y0, 0.0, 1.0), gamma)
    return torch.clamp(y, 0.0, 1.0)


def _plot_crf(fig, gs, crf_params: torch.Tensor, cam: int):
    sub = gs.subgridspec(1, 2, width_ratios=[2, 1], wspace=0.0)
    ax_plot = fig.add_subplot(sub[0])
    ax_img = fig.add_subplot(sub[1])

    cols = [(1, 0, 0, 0.6), (0, 1, 0, 0.6), (0, 0, 1, 0.6)]
    x = torch.linspace(0.0, 1.0, 256, device=crf_params.device)
    crf_cam = crf_params[cam]
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, x)
        ax_plot.plot(
            x.detach().cpu().numpy(), y.detach().cpu().numpy(),
            color=cols[ch], linewidth=2.0, label=["Red", "Green", "Blue"][ch],
        )
        center_x = float(center.detach().cpu().item())
        center_y = float(_apply_crf(toe, shoulder, gamma,
                         center, center).detach().cpu().item())
        ax_plot.scatter([center_x], [center_y], color=cols[ch],
                        s=18, zorder=6, marker="o")
    ax_plot.axvline(1.0, color="black", linestyle="--", alpha=0.5)
    ax_plot.set_xlabel("Linear Input Intensity")
    ax_plot.set_ylabel("Output Intensity")
    ax_plot.set_title("Camera Response Function (R, G, B)")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.legend()

    img = _gray_bars(256)
    xin = torch.from_numpy(
        img[img.shape[0] // 2:, :, 0].copy()).to(crf_params.device)
    xin_lin = xin.flatten()
    for ch in range(3):
        toe, shoulder, gamma, center = _crf_effective_from_raw(crf_cam[ch])
        y = _apply_crf(toe, shoulder, gamma, center, xin_lin)
        img[img.shape[0] // 2:, :,
            ch] = y.reshape(img.shape[0] // 2, img.shape[1]).cpu().numpy()
    img = _srgb_inverse_oetf(img)
    _show_image(ax_img, img, "Tone Mapping Visualization")
    size = img.shape[0]
    ax_img.text(
        size * 0.5, 20, "Linear", ha="center", va="top", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )
    ax_img.text(
        size * 0.5, size - 20, "Tone Mapped", ha="center", va="bottom", color="white",
        bbox=dict(facecolor="black", alpha=0.5),
    )


# =============================================================================
# Public API
# =============================================================================

@torch.no_grad()
def export_ppisp_report(
    ppisp: PPISP,
    frames_per_camera: list[int],
    output_dir: Path,
    camera_names: list[str] | None = None,
) -> list[Path]:
    """Generate PDF reports visualizing learned PPISP parameters.

    Creates one PDF per camera showing:
    - Exposure compensation over time
    - Vignetting curves and effect visualization
    - Color correction shifts over time (chromaticity diagram)
    - Camera response function curves

    Also exports a JSON file with all parameter values.

    Args:
        ppisp: Trained PPISP module instance.
        frames_per_camera: List of frame counts per camera.
        output_dir: Directory to write output files.
        camera_names: Optional list of camera names (defaults to "camera_0", etc.).

    Returns:
        List of paths to written PDF files.
    """
    matplotlib.use("Agg", force=True)

    num_cams = int(ppisp.num_cameras)

    if camera_names is None:
        camera_names = [f"camera_{i}" for i in range(num_cams)]

    assert len(camera_names) == num_cams, (
        f"camera_names length {len(camera_names)} != num_cameras {num_cams}"
    )
    assert len(frames_per_camera) == num_cams, (
        f"frames_per_camera length {len(frames_per_camera)} != num_cameras {num_cams}"
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    for cam in range(num_cams):
        num_rows = 5
        fig = plt.figure(figsize=(20, 5 * num_rows))
        gs = fig.add_gridspec(num_rows, 1, height_ratios=[1] * num_rows)

        row_idx = 0

        # Row 0: Exposure
        _plot_exposure(fig, gs[row_idx],
                       ppisp.exposure_params, frames_per_camera, cam)
        row_idx += 1

        # Row 1: Vignetting
        _plot_vignetting(fig, gs[row_idx], ppisp.vignetting_params, cam)
        row_idx += 1

        # Rows 2-3: Color correction
        _plot_color(fig, gs[row_idx], gs[row_idx + 1],
                    ppisp.color_params, frames_per_camera, cam)
        row_idx += 2

        # Row 4: CRF
        _plot_crf(fig, gs[row_idx], ppisp.crf_params, cam)

        plt.tight_layout()
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        out_path = output_dir / f"{safe_label}_ppisp_report.pdf"
        plt.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    # Export JSON with parameter values
    _export_params_json(ppisp, frames_per_camera, camera_names, output_dir)

    return outputs


def _export_params_json(
    ppisp: PPISP,
    frames_per_camera: list[int],
    camera_names: list[str],
    output_dir: Path,
) -> Path:
    """Export PPISP parameters to JSON."""
    num_cams = int(ppisp.num_cameras)

    # Exposure
    exposure_params_raw = ppisp.exposure_params.detach().float().cpu()
    exposure_vals = exposure_params_raw.tolist()

    # Vignetting
    vig = {}
    for cam in range(num_cams):
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        vig_cam = {}
        # Per-channel vignetting
        for ch, ch_name in enumerate(["red", "green", "blue"]):
            oc = ppisp.vignetting_params[cam, ch,
                                         :2].detach().float().cpu().tolist()
            alphas = ppisp.vignetting_params[cam,
                                             ch, 2:].detach().float().cpu().tolist()
            vig_cam[ch_name] = {"optical_center": oc, "alphas": alphas}
        vig[safe_label] = vig_cam

    # Color
    color_per_frame = []
    color_real = _color_offsets_from_params(
        ppisp.color_params).detach().float().cpu().numpy()
    for i in range(color_real.shape[0]):
        color_per_frame.append({
            "blue": color_real[i, 0].tolist(),
            "red": color_real[i, 1].tolist(),
            "green": color_real[i, 2].tolist(),
            "neutral": color_real[i, 3].tolist(),
        })

    # CRF
    crf = {}
    for cam in range(num_cams):
        cam_label = camera_names[cam] if camera_names[cam] else f"camera_{cam}"
        safe_label = "".join(c if c.isalnum() or c in (
            "-", "_") else "_" for c in cam_label)
        crf_cam = {}
        crf_cam_params = ppisp.crf_params[cam]
        for ch, ch_name in enumerate(["red", "green", "blue"]):
            raw = crf_cam_params[ch].detach().float().cpu()
            toe, shoulder, gamma, center = _crf_effective_from_raw(
                crf_cam_params[ch])
            crf_cam[ch_name] = {
                "raw": {
                    "toe_raw": float(raw[0].item()),
                    "shoulder_raw": float(raw[1].item()),
                    "gamma_raw": float(raw[2].item()),
                    "center_raw": float(raw[3].item()),
                },
                "effective": {
                    "toe": float(toe.detach().cpu().item()),
                    "shoulder": float(shoulder.detach().cpu().item()),
                    "gamma": float(gamma.detach().cpu().item()),
                    "center": float(center.detach().cpu().item()),
                },
            }
        crf[safe_label] = crf_cam

    json_obj = {
        "ppisp_version": __version__,
        "frames_per_camera": frames_per_camera,
        "exposure": {
            "per_frame": exposure_vals,
        },
        "vignetting": {"by_camera": vig},
        "color": {"per_frame": color_per_frame},
        "crf": {"by_camera": crf},
    }

    json_out = output_dir / "ppisp_params.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2)

    return json_out

```



### File: ppisp\src\ppisp_impl.cu

```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cub/cub.cuh>

#include "ppisp_math.cuh"
#include "ppisp_math_bwd.cuh"

// ============================================================================
// Configuration
// ============================================================================

// CUDA kernel launch configuration
constexpr int PPISP_BLOCK_SIZE = 256;

// Helper function to compute grid size
inline int divUp(int a, int b) { return (a + b - 1) / b; }

// ============================================================================
// PPISP Forward Kernel
// ============================================================================

__global__ void ppisp_kernel(int batch_size, int num_cameras, int num_frames,
                             const float *__restrict__ exposure_params,
                             const VignettingChannelParams *__restrict__ vignetting_params,
                             const ColorPPISPParams *__restrict__ color_params,
                             const CRFPPISPChannelParams *__restrict__ crf_params,
                             const float3 *__restrict__ rgb_in, float3 *__restrict__ rgb_out,
                             const float2 *__restrict__ pixel_coords, int resolution_x,
                             int resolution_y, int camera_idx, int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size)
        return;

    // Load RGB input
    float3 rgb = rgb_in[tid];

    // ISP Pipeline - Full PPISP

    // 1. Exposure compensation
    if (frame_idx != -1) {
        apply_exposure(rgb, exposure_params[frame_idx], rgb);
    }

    // 2. Vignetting correction
    if (camera_idx != -1) {
        apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coords[tid],
                         (float)resolution_x, (float)resolution_y, rgb);
    }

    // 3. Color correction (homography)
    if (frame_idx != -1) {
        apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb);
    }

    // 4. Camera Response Function (CRF)
    if (camera_idx != -1) {
        apply_crf_ppisp(rgb, &crf_params[camera_idx * 3], rgb);
    }

    // Store output
    rgb_out[tid] = rgb;
}

// ============================================================================
// PPISP Backward Kernel
// ============================================================================

template <int BLOCK_SIZE>
__global__ void ppisp_bwd_kernel(
    int batch_size, int num_cameras, int num_frames, const float *__restrict__ exposure_params,
    const VignettingChannelParams *__restrict__ vignetting_params,
    const ColorPPISPParams *__restrict__ color_params,
    const CRFPPISPChannelParams *__restrict__ crf_params, const float3 *__restrict__ rgb_in,
    const float3 *__restrict__ rgb_out, const float3 *__restrict__ grad_rgb_out,
    float *__restrict__ grad_exposure_params,
    VignettingChannelParams *__restrict__ grad_vignetting_params,
    ColorPPISPParams *__restrict__ grad_color_params,
    CRFPPISPChannelParams *__restrict__ grad_crf_params, float3 *__restrict__ grad_rgb_in,
    const float2 *__restrict__ pixel_coords, int resolution_x, int resolution_y, int camera_idx,
    int frame_idx) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Per-thread gradient accumulators
    float grad_exposure_local = 0.0f;
    VignettingChannelParams grad_vignetting_local[3] = {
        {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}};
    ColorPPISPParams grad_color_local = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
    CRFPPISPChannelParams grad_crf_local[3] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};

    if (tid < batch_size) {
        // Load input
        float3 rgb_input = rgb_in[tid];
        float2 pixel_coord = pixel_coords[tid];

        // Recompute forward pass using separate output variables to avoid aliasing
        float3 rgb = rgb_input;
        float3 rgb_after_exp = rgb;
        float3 rgb_after_vig = rgb;
        float3 rgb_after_color = rgb;

        // 1. Exposure
        if (frame_idx != -1) {
            apply_exposure(rgb, exposure_params[frame_idx], rgb_after_exp);
            rgb = rgb_after_exp;
        }

        // 2. Vignetting
        if (camera_idx != -1) {
            apply_vignetting(rgb, &vignetting_params[camera_idx * 3], pixel_coord,
                             (float)resolution_x, (float)resolution_y, rgb_after_vig);
            rgb = rgb_after_vig;
        } else {
            rgb_after_vig = rgb;
        }

        // 3. Color correction
        if (frame_idx != -1) {
            apply_color_correction_ppisp(rgb, &color_params[frame_idx], rgb_after_color);
            rgb = rgb_after_color;
        } else {
            rgb_after_color = rgb;
        }

        // Backward pass (reverse order)
        float3 grad_rgb = grad_rgb_out[tid];

        // 4. CRF backward
        if (camera_idx != -1) {
            apply_crf_ppisp_bwd(rgb_after_color, &crf_params[camera_idx * 3], grad_rgb, grad_rgb,
                                grad_crf_local);
        }

        // 3. Color correction backward
        if (frame_idx != -1) {
            apply_color_correction_ppisp_bwd(rgb_after_vig, &color_params[frame_idx], grad_rgb,
                                             grad_rgb, &grad_color_local);
        }

        // 2. Vignetting backward
        if (camera_idx != -1) {
            apply_vignetting_bwd(rgb_after_exp, &vignetting_params[camera_idx * 3], pixel_coord,
                                 (float)resolution_x, (float)resolution_y, grad_rgb, grad_rgb,
                                 grad_vignetting_local);
        }

        // 1. Exposure backward
        if (frame_idx != -1) {
            apply_exposure_bwd(rgb_input, exposure_params[frame_idx], grad_rgb, grad_rgb,
                               grad_exposure_local);
        }

        // Store RGB input gradient
        grad_rgb_in[tid] = grad_rgb;
    }  // END if (tid < batch_size)

    // Block-level reduction and atomic add for parameter gradients
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduceFloat;
    typedef cub::BlockReduce<float2, BLOCK_SIZE> BlockReduceFloat2;

    if (frame_idx != -1) {
        // Exposure
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            float val = BlockReduceFloat(temp).Sum(grad_exposure_local);
            if (threadIdx.x == 0)
                atomicAdd(&grad_exposure_params[frame_idx], val);
        }

        // Color params (4 x float2)
        {
            __shared__ typename BlockReduceFloat2::TempStorage temp;
            ColorPPISPParams *grad_color_out = &grad_color_params[frame_idx];

            float2 val_b = BlockReduceFloat2(temp).Sum(grad_color_local.b);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->b.x, val_b.x);
                atomicAdd(&grad_color_out->b.y, val_b.y);
            }

            float2 val_r = BlockReduceFloat2(temp).Sum(grad_color_local.r);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->r.x, val_r.x);
                atomicAdd(&grad_color_out->r.y, val_r.y);
            }

            float2 val_g = BlockReduceFloat2(temp).Sum(grad_color_local.g);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->g.x, val_g.x);
                atomicAdd(&grad_color_out->g.y, val_g.y);
            }

            float2 val_n = BlockReduceFloat2(temp).Sum(grad_color_local.n);
            __syncthreads();
            if (threadIdx.x == 0) {
                atomicAdd(&grad_color_out->n.x, val_n.x);
                atomicAdd(&grad_color_out->n.y, val_n.y);
            }
        }
    }

    if (camera_idx != -1) {
        // Vignetting params (3 channels x 5 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            VignettingChannelParams *grad_vig_out = &grad_vignetting_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_cx = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cx);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cx, val_cx);

                float val_cy = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].cy);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].cy, val_cy);

                float val_a0 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha0);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha0, val_a0);

                float val_a1 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha1);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha1, val_a1);

                float val_a2 = BlockReduceFloat(temp).Sum(grad_vignetting_local[ch].alpha2);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_vig_out[ch].alpha2, val_a2);
            }
        }

        // CRF params (3 channels x 4 params)
        {
            __shared__ typename BlockReduceFloat::TempStorage temp;
            CRFPPISPChannelParams *grad_crf_out = &grad_crf_params[camera_idx * 3];

#pragma unroll
            for (int ch = 0; ch < 3; ch++) {
                float val_toe = BlockReduceFloat(temp).Sum(grad_crf_local[ch].toe);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].toe, val_toe);

                float val_shoulder = BlockReduceFloat(temp).Sum(grad_crf_local[ch].shoulder);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].shoulder, val_shoulder);

                float val_gamma = BlockReduceFloat(temp).Sum(grad_crf_local[ch].gamma);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].gamma, val_gamma);

                float val_center = BlockReduceFloat(temp).Sum(grad_crf_local[ch].center);
                __syncthreads();
                if (threadIdx.x == 0)
                    atomicAdd(&grad_crf_out[ch].center, val_center);
            }
        }
    }
}

// ============================================================================
// Forward Pass Implementation
// ============================================================================

void ppisp_forward(const float *exposure_params, const float *vignetting_params,
                   const float *color_params, const float *crf_params, const float *rgb_in,
                   float *rgb_out, const float *pixel_coords, int num_pixels, int num_cameras,
                   int num_frames, int resolution_w, int resolution_h, int camera_idx,
                   int frame_idx) {
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);

    ppisp_kernel<<<blocks, threads>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<float3 *>(rgb_out),
        reinterpret_cast<const float2 *>(pixel_coords), resolution_w, resolution_h, camera_idx,
        frame_idx);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in ppisp_forward: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================================
// Backward Pass Implementation
// ============================================================================

void ppisp_backward(const float *exposure_params, const float *vignetting_params,
                    const float *color_params, const float *crf_params, const float *rgb_in,
                    const float *rgb_out, const float *pixel_coords, const float *v_rgb_out,
                    float *v_exposure_params, float *v_vignetting_params, float *v_color_params,
                    float *v_crf_params, float *v_rgb_in, int num_pixels, int num_cameras,
                    int num_frames, int resolution_w, int resolution_h, int camera_idx,
                    int frame_idx) {
    const int threads = PPISP_BLOCK_SIZE;
    const int blocks = divUp(num_pixels, threads);

    ppisp_bwd_kernel<PPISP_BLOCK_SIZE><<<blocks, threads>>>(
        num_pixels, num_cameras, num_frames, exposure_params,
        reinterpret_cast<const VignettingChannelParams *>(vignetting_params),
        reinterpret_cast<const ColorPPISPParams *>(color_params),
        reinterpret_cast<const CRFPPISPChannelParams *>(crf_params),
        reinterpret_cast<const float3 *>(rgb_in), reinterpret_cast<const float3 *>(rgb_out),
        reinterpret_cast<const float3 *>(v_rgb_out), v_exposure_params,
        reinterpret_cast<VignettingChannelParams *>(v_vignetting_params),
        reinterpret_cast<ColorPPISPParams *>(v_color_params),
        reinterpret_cast<CRFPPISPChannelParams *>(v_crf_params),
        reinterpret_cast<float3 *>(v_rgb_in), reinterpret_cast<const float2 *>(pixel_coords),
        resolution_w, resolution_h, camera_idx, frame_idx);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in ppisp_backward: %s\n", cudaGetErrorString(err));
    }
}

```



### File: ppisp\src\ppisp_math.cuh

```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

// ============================================================================
// Note: make_float2, make_float3, make_float4 are provided by CUDA
// ============================================================================

// ============================================================================
// Matrix Types (row-major storage for efficient access)
// ============================================================================

// 2x2 matrix (4 floats, row-major)
struct float2x2 {
    float m[4];  // [m00, m01, m10, m11]

    __device__ __forceinline__ float &operator()(int row, int col) { return m[row * 2 + col]; }

    __device__ __forceinline__ const float &operator()(int row, int col) const {
        return m[row * 2 + col];
    }
};

// 3x3 matrix (9 floats, row-major)
struct float3x3 {
    float m[9];  // [m00, m01, m02, m10, m11, m12, m20, m21, m22]

    __device__ __forceinline__ float &operator()(int row, int col) { return m[row * 3 + col]; }

    __device__ __forceinline__ const float &operator()(int row, int col) const {
        return m[row * 3 + col];
    }
};

// 4x4 matrix (16 floats, row-major)
struct float4x4 {
    float m[16];  // row-major

    __device__ __forceinline__ float &operator()(int row, int col) { return m[row * 4 + col]; }

    __device__ __forceinline__ const float &operator()(int row, int col) const {
        return m[row * 4 + col];
    }
};

// Matrix constructors
__device__ __forceinline__ float2x2 make_float2x2(float m00, float m01, float m10, float m11) {
    float2x2 mat;
    mat.m[0] = m00;
    mat.m[1] = m01;
    mat.m[2] = m10;
    mat.m[3] = m11;
    return mat;
}

__device__ __forceinline__ float3x3 make_float3x3(float m00, float m01, float m02, float m10,
                                                  float m11, float m12, float m20, float m21,
                                                  float m22) {
    float3x3 mat;
    mat.m[0] = m00;
    mat.m[1] = m01;
    mat.m[2] = m02;
    mat.m[3] = m10;
    mat.m[4] = m11;
    mat.m[5] = m12;
    mat.m[6] = m20;
    mat.m[7] = m21;
    mat.m[8] = m22;
    return mat;
}

__device__ __forceinline__ float4x4 make_float4x4(float m00, float m01, float m02, float m03,
                                                  float m10, float m11, float m12, float m13,
                                                  float m20, float m21, float m22, float m23,
                                                  float m30, float m31, float m32, float m33) {
    float4x4 mat;
    mat.m[0] = m00;
    mat.m[1] = m01;
    mat.m[2] = m02;
    mat.m[3] = m03;
    mat.m[4] = m10;
    mat.m[5] = m11;
    mat.m[6] = m12;
    mat.m[7] = m13;
    mat.m[8] = m20;
    mat.m[9] = m21;
    mat.m[10] = m22;
    mat.m[11] = m23;
    mat.m[12] = m30;
    mat.m[13] = m31;
    mat.m[14] = m32;
    mat.m[15] = m33;
    return mat;
}

// ============================================================================
// Vector Operators (device-only, optimized with intrinsics)
// ============================================================================

// float2 operators
__device__ __forceinline__ float2 operator+(const float2 &a, const float2 &b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 operator-(const float2 &a, const float2 &b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 operator*(const float2 &a, float s) {
    return make_float2(a.x * s, a.y * s);
}

__device__ __forceinline__ float2 operator*(float s, const float2 &a) {
    return make_float2(a.x * s, a.y * s);
}

__device__ __forceinline__ float2 operator*(const float2 &a, const float2 &b) {
    return make_float2(a.x * b.x, a.y * b.y);
}

__device__ __forceinline__ float2 operator/(const float2 &a, float s) {
    float inv_s = __fdividef(1.0f, s);
    return make_float2(a.x * inv_s, a.y * inv_s);
}

// float3 operators
__device__ __forceinline__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __forceinline__ float3 operator*(const float3 &a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(float s, const float3 &a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ __forceinline__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __forceinline__ float3 operator/(const float3 &a, float s) {
    float inv_s = __fdividef(1.0f, s);
    return make_float3(a.x * inv_s, a.y * inv_s, a.z * inv_s);
}

// float4 operators
__device__ __forceinline__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ __forceinline__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ __forceinline__ float4 operator*(const float4 &a, float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __forceinline__ float4 operator*(float s, const float4 &a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

__device__ __forceinline__ float4 operator*(const float4 &a, const float4 &b) {
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

__device__ __forceinline__ float4 operator/(const float4 &a, float s) {
    float inv_s = __fdividef(1.0f, s);
    return make_float4(a.x * inv_s, a.y * inv_s, a.z * inv_s, a.w * inv_s);
}

// ============================================================================
// Vector Functions (optimized)
// ============================================================================

// Dot product (using FMA for precision)
__device__ __forceinline__ float dot(const float2 &a, const float2 &b) {
    return __fmaf_rn(a.x, b.x, a.y * b.y);
}

__device__ __forceinline__ float dot(const float3 &a, const float3 &b) {
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, a.z * b.z));
}

__device__ __forceinline__ float dot(const float4 &a, const float4 &b) {
    return __fmaf_rn(a.x, b.x, __fmaf_rn(a.y, b.y, __fmaf_rn(a.z, b.z, a.w * b.w)));
}

// Cross product
__device__ __forceinline__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// Length
__device__ __forceinline__ float length(const float2 &a) { return sqrtf(dot(a, a)); }

__device__ __forceinline__ float length(const float3 &a) { return sqrtf(dot(a, a)); }

__device__ __forceinline__ float length(const float4 &a) { return sqrtf(dot(a, a)); }

// Fast reciprocal length (less precise but faster)
__device__ __forceinline__ float rlength(const float2 &a) { return __frsqrt_rn(dot(a, a)); }

__device__ __forceinline__ float rlength(const float3 &a) { return __frsqrt_rn(dot(a, a)); }

__device__ __forceinline__ float rlength(const float4 &a) { return __frsqrt_rn(dot(a, a)); }

// Normalize
__device__ __forceinline__ float2 normalize(const float2 &a) { return a * rlength(a); }

__device__ __forceinline__ float3 normalize(const float3 &a) { return a * rlength(a); }

__device__ __forceinline__ float4 normalize(const float4 &a) { return a * rlength(a); }

// Clamp (per-component)
__device__ __forceinline__ float2 clamp(const float2 &v, float min_val, float max_val) {
    return make_float2(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val));
}

__device__ __forceinline__ float3 clamp(const float3 &v, float min_val, float max_val) {
    return make_float3(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val),
                       fminf(fmaxf(v.z, min_val), max_val));
}

__device__ __forceinline__ float4 clamp(const float4 &v, float min_val, float max_val) {
    return make_float4(fminf(fmaxf(v.x, min_val), max_val), fminf(fmaxf(v.y, min_val), max_val),
                       fminf(fmaxf(v.z, min_val), max_val), fminf(fmaxf(v.w, min_val), max_val));
}

// Lerp (linear interpolation) - optimized with FMA
__device__ __forceinline__ float lerp(float a, float b, float t) { return __fmaf_rn(b - a, t, a); }

__device__ __forceinline__ float2 lerp(const float2 &a, const float2 &b, float t) {
    return make_float2(lerp(a.x, b.x, t), lerp(a.y, b.y, t));
}

__device__ __forceinline__ float3 lerp(const float3 &a, const float3 &b, float t) {
    return make_float3(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t));
}

__device__ __forceinline__ float4 lerp(const float4 &a, const float4 &b, float t) {
    return make_float4(lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t), lerp(a.w, b.w, t));
}

// Element-wise pow (use __powf for speed)
__device__ __forceinline__ float2 pow(const float2 &v, float exp) {
    return make_float2(__powf(v.x, exp), __powf(v.y, exp));
}

__device__ __forceinline__ float3 pow(const float3 &v, float exp) {
    return make_float3(__powf(v.x, exp), __powf(v.y, exp), __powf(v.z, exp));
}

__device__ __forceinline__ float3 pow(const float3 &v, const float3 &exp) {
    return make_float3(__powf(v.x, exp.x), __powf(v.y, exp.y), __powf(v.z, exp.z));
}

__device__ __forceinline__ float4 pow(const float4 &v, float exp) {
    return make_float4(__powf(v.x, exp), __powf(v.y, exp), __powf(v.z, exp), __powf(v.w, exp));
}

// Min/Max (per-component)
__device__ __forceinline__ float2 min(const float2 &a, const float2 &b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__device__ __forceinline__ float3 min(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float4 min(const float4 &a, const float4 &b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

__device__ __forceinline__ float2 max(const float2 &a, const float2 &b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__device__ __forceinline__ float3 max(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float4 max(const float4 &a, const float4 &b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

// ============================================================================
// Matrix-Vector Operations (optimized with FMA)
// ============================================================================

// 2x2 matrix-vector multiplication: y = A * x
__device__ __forceinline__ float2 operator*(const float2x2 &mat, const float2 &v) {
    return make_float2(__fmaf_rn(mat.m[0], v.x, mat.m[1] * v.y),
                       __fmaf_rn(mat.m[2], v.x, mat.m[3] * v.y));
}

// 3x3 matrix-vector multiplication: y = A * x
__device__ __forceinline__ float3 operator*(const float3x3 &mat, const float3 &v) {
    return make_float3(__fmaf_rn(mat.m[0], v.x, __fmaf_rn(mat.m[1], v.y, mat.m[2] * v.z)),
                       __fmaf_rn(mat.m[3], v.x, __fmaf_rn(mat.m[4], v.y, mat.m[5] * v.z)),
                       __fmaf_rn(mat.m[6], v.x, __fmaf_rn(mat.m[7], v.y, mat.m[8] * v.z)));
}

// 4x4 matrix-vector multiplication: y = A * x
__device__ __forceinline__ float4 operator*(const float4x4 &mat, const float4 &v) {
    return make_float4(
        __fmaf_rn(mat.m[0], v.x,
                  __fmaf_rn(mat.m[1], v.y, __fmaf_rn(mat.m[2], v.z, mat.m[3] * v.w))),
        __fmaf_rn(mat.m[4], v.x,
                  __fmaf_rn(mat.m[5], v.y, __fmaf_rn(mat.m[6], v.z, mat.m[7] * v.w))),
        __fmaf_rn(mat.m[8], v.x,
                  __fmaf_rn(mat.m[9], v.y, __fmaf_rn(mat.m[10], v.z, mat.m[11] * v.w))),
        __fmaf_rn(mat.m[12], v.x,
                  __fmaf_rn(mat.m[13], v.y, __fmaf_rn(mat.m[14], v.z, mat.m[15] * v.w))));
}

// ============================================================================
// Matrix-Matrix Operations (optimized with FMA and unrolling)
// ============================================================================

// 2x2 matrix multiplication: C = A * B
__device__ __forceinline__ float2x2 operator*(const float2x2 &A, const float2x2 &B) {
    float2x2 C;
#pragma unroll
    for (int i = 0; i < 2; i++) {
#pragma unroll
        for (int j = 0; j < 2; j++) {
            C(i, j) = __fmaf_rn(A(i, 0), B(0, j), A(i, 1) * B(1, j));
        }
    }
    return C;
}

// 3x3 matrix multiplication: C = A * B
__device__ __forceinline__ float3x3 operator*(const float3x3 &A, const float3x3 &B) {
    float3x3 C;
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            C(i, j) = __fmaf_rn(A(i, 0), B(0, j), __fmaf_rn(A(i, 1), B(1, j), A(i, 2) * B(2, j)));
        }
    }
    return C;
}

// Matrix transpose
__device__ __forceinline__ float2x2 transpose(const float2x2 &A) {
    float2x2 AT;
    AT(0, 0) = A(0, 0);
    AT(0, 1) = A(1, 0);
    AT(1, 0) = A(0, 1);
    AT(1, 1) = A(1, 1);
    return AT;
}

__device__ __forceinline__ float3x3 transpose(const float3x3 &A) {
    float3x3 AT;
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            AT(i, j) = A(j, i);
        }
    }
    return AT;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Identity matrices
__device__ __forceinline__ float2x2 identity_2x2() { return make_float2x2(1.0f, 0.0f, 0.0f, 1.0f); }

__device__ __forceinline__ float3x3 identity_3x3() {
    return make_float3x3(1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

// ============================================================================
// ISP Parameter Structures
// ============================================================================

// Vignetting parameters per channel (5 params: cx, cy, alpha0, alpha1, alpha2)
struct VignettingChannelParams {
    float cx;      // Optical center X
    float cy;      // Optical center Y
    float alpha0;  // r^2 coefficient
    float alpha1;  // r^4 coefficient
    float alpha2;  // r^6 coefficient
};

// CRF parameters per channel for PPISP (4 params, pre-transformation)
struct CRFPPISPChannelParams {
    float toe;       // Toe strength (before softplus)
    float shoulder;  // Shoulder strength (before softplus)
    float gamma;     // Gamma (before softplus)
    float center;    // Center point (before sigmoid)
};

// Color correction parameters for PPISP (4 color points: BRGN, 2D offsets each)
struct ColorPPISPParams {
    float2 b;  // Blue latent offsets
    float2 r;  // Red latent offsets
    float2 g;  // Green latent offsets
    float2 n;  // Neutral latent offsets
};

// ============================================================================
// ISP-Specific Helper Functions
// ============================================================================

// Color correction pinv blocks (constant memory)
__constant__ float COLOR_PINV_BLOCKS[4][4] = {
    {0.0480542f, -0.0043631f, -0.0043631f, 0.0481283f},  // Blue
    {0.0580570f, -0.0179872f, -0.0179872f, 0.0431061f},  // Red
    {0.0433336f, -0.0180537f, -0.0180537f, 0.0580500f},  // Green
    {0.0128369f, -0.0034654f, -0.0034654f, 0.0128158f}   // Neutral
};

// Softplus transformation for bounded positive parameters
__device__ __forceinline__ float bounded_positive_forward(float raw, float min_value = 0.1f) {
    return min_value + __logf(1.0f + __expf(raw));
}

// Sigmoid transformation for clamped parameters
__device__ __forceinline__ float clamped_forward(float raw) {
    return __fdividef(1.0f, 1.0f + __expf(-raw));
}

// Compute 3x3 homography matrix from color parameters
__device__ __forceinline__ float3x3 compute_homography(const ColorPPISPParams *params) {
    // Load latent offsets for control chromaticities (B, R, G, N)
    const float2 &b_lat = params->b;
    const float2 &r_lat = params->r;
    const float2 &g_lat = params->g;
    const float2 &n_lat = params->n;

    // Map latent to real offsets via ZCA 2x2 blocks (stored as constant 4-element arrays)
    float2x2 zca_b, zca_r, zca_g, zca_n;
    zca_b.m[0] = COLOR_PINV_BLOCKS[0][0];
    zca_b.m[1] = COLOR_PINV_BLOCKS[0][1];
    zca_b.m[2] = COLOR_PINV_BLOCKS[0][2];
    zca_b.m[3] = COLOR_PINV_BLOCKS[0][3];

    zca_r.m[0] = COLOR_PINV_BLOCKS[1][0];
    zca_r.m[1] = COLOR_PINV_BLOCKS[1][1];
    zca_r.m[2] = COLOR_PINV_BLOCKS[1][2];
    zca_r.m[3] = COLOR_PINV_BLOCKS[1][3];

    zca_g.m[0] = COLOR_PINV_BLOCKS[2][0];
    zca_g.m[1] = COLOR_PINV_BLOCKS[2][1];
    zca_g.m[2] = COLOR_PINV_BLOCKS[2][2];
    zca_g.m[3] = COLOR_PINV_BLOCKS[2][3];

    zca_n.m[0] = COLOR_PINV_BLOCKS[3][0];
    zca_n.m[1] = COLOR_PINV_BLOCKS[3][1];
    zca_n.m[2] = COLOR_PINV_BLOCKS[3][2];
    zca_n.m[3] = COLOR_PINV_BLOCKS[3][3];

    float2 bd = zca_b * b_lat;
    float2 rd = zca_r * r_lat;
    float2 gd = zca_g * g_lat;
    float2 nd = zca_n * n_lat;

    // Fixed sources (r, g, intensity) + offsets = targets
    float3 t_b = make_float3(0.0f + bd.x, 0.0f + bd.y, 1.0f);
    float3 t_r = make_float3(1.0f + rd.x, 0.0f + rd.y, 1.0f);
    float3 t_g = make_float3(0.0f + gd.x, 1.0f + gd.y, 1.0f);
    float3 t_gray = make_float3(1.0f / 3.0f + nd.x, 1.0f / 3.0f + nd.y, 1.0f);

    // T has columns [t_b, t_r, t_g] (column-major stored as row-major)
    float3x3 T = make_float3x3(t_b.x, t_r.x, t_g.x, t_b.y, t_r.y, t_g.y, t_b.z, t_r.z, t_g.z);

    // Skew-symmetric matrix [t_gray]_x
    float3x3 skew = make_float3x3(0.0f, -t_gray.z, t_gray.y, t_gray.z, 0.0f, -t_gray.x, -t_gray.y,
                                  t_gray.x, 0.0f);

    // M = skew * T
    float3x3 M = skew * T;

    // Nullspace vector lambda via cross product of rows
    float3 r0 = make_float3(M(0, 0), M(0, 1), M(0, 2));
    float3 r1 = make_float3(M(1, 0), M(1, 1), M(1, 2));
    float3 r2 = make_float3(M(2, 0), M(2, 1), M(2, 2));

    float3 lambda_v = cross(r0, r1);
    float n2 = dot(lambda_v, lambda_v);

    if (n2 < 1.0e-20f) {
        lambda_v = cross(r0, r2);
        n2 = dot(lambda_v, lambda_v);
        if (n2 < 1.0e-20f) {
            lambda_v = cross(r1, r2);
        }
    }

    // S_inv = [[-1,-1,1],[1,0,0],[0,1,0]]
    float3x3 S_inv = make_float3x3(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    // D = diag(lambda)
    float3x3 D =
        make_float3x3(lambda_v.x, 0.0f, 0.0f, 0.0f, lambda_v.y, 0.0f, 0.0f, 0.0f, lambda_v.z);

    // H = T * D * S_inv
    float3x3 H = T * D * S_inv;

    // Normalize so H[2][2] = 1
    float s = H(2, 2);
    if (fabsf(s) > 1.0e-20f) {
        float inv_s = 1.0f / s;
#pragma unroll
        for (int i = 0; i < 9; i++) {
            H.m[i] *= inv_s;
        }
    }

    return H;
}

// ----------------------------------------------------------------------------
// 3x3 Matrix Inverse
// ----------------------------------------------------------------------------
__device__ __forceinline__ float3x3 matrix_inverse_3x3(const float3x3 &M) {
    // Compute determinant
    float det = M.m[0] * (M.m[4] * M.m[8] - M.m[5] * M.m[7]) -
                M.m[1] * (M.m[3] * M.m[8] - M.m[5] * M.m[6]) +
                M.m[2] * (M.m[3] * M.m[7] - M.m[4] * M.m[6]);

    // Add epsilon to prevent division by zero
    const float eps = 1e-8f;
    float inv_det = 1.0f / (fabsf(det) > eps ? det : (det >= 0.0f ? eps : -eps));

    // Compute cofactor matrix and transpose (adjugate)
    float3x3 inv;
    inv.m[0] = (M.m[4] * M.m[8] - M.m[5] * M.m[7]) * inv_det;
    inv.m[1] = (M.m[2] * M.m[7] - M.m[1] * M.m[8]) * inv_det;
    inv.m[2] = (M.m[1] * M.m[5] - M.m[2] * M.m[4]) * inv_det;
    inv.m[3] = (M.m[5] * M.m[6] - M.m[3] * M.m[8]) * inv_det;
    inv.m[4] = (M.m[0] * M.m[8] - M.m[2] * M.m[6]) * inv_det;
    inv.m[5] = (M.m[2] * M.m[3] - M.m[0] * M.m[5]) * inv_det;
    inv.m[6] = (M.m[3] * M.m[7] - M.m[4] * M.m[6]) * inv_det;
    inv.m[7] = (M.m[1] * M.m[6] - M.m[0] * M.m[7]) * inv_det;
    inv.m[8] = (M.m[0] * M.m[4] - M.m[1] * M.m[3]) * inv_det;

    return inv;
}

// ============================================================================
// Modular ISP Component Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Exposure Compensation
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_exposure(const float3 &rgb_in, float exposure_param,
                                               float3 &rgb_out) {
    float exposure_factor = exp2f(exposure_param);
    rgb_out = rgb_in * exposure_factor;
}

// ----------------------------------------------------------------------------
// Vignetting Correction
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_vignetting(const float3 &rgb_in,
                                                 const VignettingChannelParams *vignetting_params,
                                                 const float2 &pixel_coords, float resolution_x,
                                                 float resolution_y, float3 &rgb_out) {
    // Normalize coordinates to [-0.5, 0.5] range based on max dimension
    float max_res = fmaxf(resolution_x, resolution_y);
    float2 uv = make_float2(__fdividef(pixel_coords.x - resolution_x * 0.5f, max_res),
                            __fdividef(pixel_coords.y - resolution_y * 0.5f, max_res));

    float rgb_arr[3] = {rgb_in.x, rgb_in.y, rgb_in.z};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const VignettingChannelParams &params = vignetting_params[i];

        float dx = uv.x - params.cx;
        float dy = uv.y - params.cy;
        float r2 = __fmaf_rn(dx, dx, dy * dy);
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float falloff = __fmaf_rn(params.alpha2, r6,
                                  __fmaf_rn(params.alpha1, r4, __fmaf_rn(params.alpha0, r2, 1.0f)));
        falloff = fmaxf(0.0f, fminf(1.0f, falloff));

        rgb_arr[i] *= falloff;
    }

    rgb_out = make_float3(rgb_arr[0], rgb_arr[1], rgb_arr[2]);
}

// ----------------------------------------------------------------------------
// Color Correction - PPISP (Homography)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_color_correction_ppisp(const float3 &rgb_in,
                                                             const ColorPPISPParams *color_params,
                                                             float3 &rgb_out) {
    // Compute homography matrix from control-point parametrization
    float3x3 H = compute_homography(color_params);

    // Convert to RGI space
    float intensity = rgb_in.x + rgb_in.y + rgb_in.z;
    float3 rgi_in = make_float3(rgb_in.x, rgb_in.y, intensity);

    // Apply homography
    float3 rgi_out = H * rgi_in;

    // Normalize and convert back to RGB
    float norm_factor = __fdividef(intensity, rgi_out.z + 1.0e-5f);
    rgi_out = rgi_out * norm_factor;

    rgb_out = make_float3(rgi_out.x, rgi_out.y, rgi_out.z - rgi_out.x - rgi_out.y);
}

// ----------------------------------------------------------------------------
// CRF - PPISP (Parametric toe-shoulder curve)
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_crf_ppisp(const float3 &rgb_in,
                                                const CRFPPISPChannelParams *crf_params,
                                                float3 &rgb_out) {
    float3 rgb_clamped = clamp(rgb_in, 0.0f, 1.0f);
    float out_arr[3];

#pragma unroll
    for (int i = 0; i < 3; i++) {
        float x = (i == 0) ? rgb_clamped.x : (i == 1) ? rgb_clamped.y : rgb_clamped.z;

        const CRFPPISPChannelParams &params = crf_params[i];

        // Transform parameters
        float toe = bounded_positive_forward(params.toe, 0.3f);
        float shoulder = bounded_positive_forward(params.shoulder, 0.3f);
        float gamma = bounded_positive_forward(params.gamma, 0.1f);
        float center = clamped_forward(params.center);

        // Compute a, b coefficients
        float lerp_val = __fmaf_rn(shoulder - toe, center, toe);
        float a = __fdividef(shoulder * center, lerp_val);
        float b = 1.0f - a;

        float y;

        // Piecewise toe-shoulder curve
        if (x <= center) {
            y = a * __powf(__fdividef(x, center), toe);
        } else {
            y = 1.0f - b * __powf(__fdividef(1.0f - x, 1.0f - center), shoulder);
        }

        // Apply gamma
        out_arr[i] = __powf(fmaxf(0.0f, y), gamma);
    }

    rgb_out = make_float3(out_arr[0], out_arr[1], out_arr[2]);
}

```



### File: ppisp\src\ppisp_math_bwd.cuh

```
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>

#include "ppisp_math.cuh"

// ============================================================================
// Backward Pass Helpers for Vector Operations
// ============================================================================

// Dot product backward: d_a, d_b given grad_output
// forward: out = dot(a, b) = a.x*b.x + a.y*b.y
// backward: d_a = grad_output * b, d_b = grad_output * a
__device__ __forceinline__ void dot_bwd(float grad_output, const float2 &a, const float2 &b,
                                        float2 &grad_a, float2 &grad_b) {
    grad_a = grad_output * b;
    grad_b = grad_output * a;
}

__device__ __forceinline__ void dot_bwd(float grad_output, const float3 &a, const float3 &b,
                                        float3 &grad_a, float3 &grad_b) {
    grad_a = grad_output * b;
    grad_b = grad_output * a;
}

__device__ __forceinline__ void dot_bwd(float grad_output, const float4 &a, const float4 &b,
                                        float4 &grad_a, float4 &grad_b) {
    grad_a = grad_output * b;
    grad_b = grad_output * a;
}

// Cross product backward
// forward: out = cross(a, b) = a x b
// backward: grad_a = b x grad_out, grad_b = grad_out x a
__device__ __forceinline__ void cross_bwd(const float3 &grad_output, const float3 &a,
                                          const float3 &b, float3 &grad_a, float3 &grad_b) {
    grad_a = cross(b, grad_output);
    grad_b = cross(grad_output, a);
}

// Length backward
// forward: out = sqrt(dot(a, a))
// backward: d_a = grad_output * a / length(a)
__device__ __forceinline__ void length_bwd(float grad_output, const float2 &a, float2 &grad_a) {
    float len = length(a);
    grad_a = (grad_output / (len + 1e-8f)) * a;
}

__device__ __forceinline__ void length_bwd(float grad_output, const float3 &a, float3 &grad_a) {
    float len = length(a);
    grad_a = (grad_output / (len + 1e-8f)) * a;
}

// Normalize backward
// forward: out = a / length(a)
// backward: d_a = (grad_out - dot(grad_out, out) * out) / length(a)
__device__ __forceinline__ void normalize_bwd(const float2 &grad_output, const float2 &a,
                                              const float2 &normalized_a, float2 &grad_a) {
    float len = length(a);
    float dot_val = dot(grad_output, normalized_a);
    grad_a = (grad_output - dot_val * normalized_a) / (len + 1e-8f);
}

__device__ __forceinline__ void normalize_bwd(const float3 &grad_output, const float3 &a,
                                              const float3 &normalized_a, float3 &grad_a) {
    float len = length(a);
    float dot_val = dot(grad_output, normalized_a);
    grad_a = (grad_output - dot_val * normalized_a) / (len + 1e-8f);
}

// ============================================================================
// Backward Pass Helpers for Matrix Operations
// ============================================================================

// Matrix-vector multiplication backward: y = A * x
// Given: grad_y (gradient w.r.t. output y)
// Compute: grad_A, grad_x
__device__ __forceinline__ void mul_bwd(const float2x2 &A, const float2 &x, const float2 &grad_y,
                                        float2x2 &grad_A, float2 &grad_x) {
    // grad_x = A^T * grad_y
    grad_x = transpose(A) * grad_y;

    // grad_A = grad_y x x^T (outer product)
#pragma unroll
    for (int i = 0; i < 2; i++) {
        float grad_y_i = (i == 0) ? grad_y.x : grad_y.y;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            float x_j = (j == 0) ? x.x : x.y;
            grad_A(i, j) = grad_y_i * x_j;
        }
    }
}

__device__ __forceinline__ void mul_bwd(const float3x3 &A, const float3 &x, const float3 &grad_y,
                                        float3x3 &grad_A, float3 &grad_x) {
    // grad_x = A^T * grad_y
    grad_x = transpose(A) * grad_y;

    // grad_A = grad_y x x^T (outer product)
#pragma unroll
    for (int i = 0; i < 3; i++) {
        float grad_y_i = (i == 0) ? grad_y.x : (i == 1) ? grad_y.y : grad_y.z;
#pragma unroll
        for (int j = 0; j < 3; j++) {
            float x_j = (j == 0) ? x.x : (j == 1) ? x.y : x.z;
            grad_A(i, j) = grad_y_i * x_j;
        }
    }
}

// Matrix-matrix multiplication backward: C = A * B
// Given: grad_C (gradient w.r.t. output C)
// Compute: grad_A, grad_B
__device__ __forceinline__ void mul_mat_bwd(const float2x2 &A, const float2x2 &B,
                                            const float2x2 &grad_C, float2x2 &grad_A,
                                            float2x2 &grad_B) {
    // grad_A = grad_C * B^T
    grad_A = grad_C * transpose(B);

    // grad_B = A^T * grad_C
    grad_B = transpose(A) * grad_C;
}

__device__ __forceinline__ void mul_mat_bwd(const float3x3 &A, const float3x3 &B,
                                            const float3x3 &grad_C, float3x3 &grad_A,
                                            float3x3 &grad_B) {
    // grad_A = grad_C * B^T
    grad_A = grad_C * transpose(B);

    // grad_B = A^T * grad_C
    grad_B = transpose(A) * grad_C;
}

// Transpose backward (transpose is self-adjoint)
// Given: grad_AT (gradient w.r.t. output A^T)
// Compute: grad_A
__device__ __forceinline__ float2x2 transpose_bwd(const float2x2 &grad_AT) {
    return transpose(grad_AT);
}

__device__ __forceinline__ float3x3 transpose_bwd(const float3x3 &grad_AT) {
    return transpose(grad_AT);
}

// ============================================================================
// Backward Pass Helpers for Scalar Operations
// ============================================================================

// Clamp backward (per-component)
// forward: out = clamp(x, min, max)
// backward: d_x = grad_out if min <= x <= max else 0
__device__ __forceinline__ void clamp_bwd(const float3 &grad_output, const float3 &x, float min_val,
                                          float max_val, float3 &grad_x) {
    grad_x.x = (x.x > min_val && x.x < max_val) ? grad_output.x : 0.0f;
    grad_x.y = (x.y > min_val && x.y < max_val) ? grad_output.y : 0.0f;
    grad_x.z = (x.z > min_val && x.z < max_val) ? grad_output.z : 0.0f;
}

// Lerp backward
// forward: out = a + (b - a) * t
// backward: d_a = grad_out * (1 - t), d_b = grad_out * t, d_t = grad_out *
// (b - a)
__device__ __forceinline__ void lerp_bwd(float grad_output, float a, float b, float t,
                                         float &grad_a, float &grad_b, float &grad_t) {
    grad_a = grad_output * (1.0f - t);
    grad_b = grad_output * t;
    grad_t = grad_output * (b - a);
}

__device__ __forceinline__ void lerp_bwd(const float3 &grad_output, const float3 &a,
                                         const float3 &b, float t, float3 &grad_a, float3 &grad_b,
                                         float &grad_t) {
    grad_a = grad_output * (1.0f - t);
    grad_b = grad_output * t;
    grad_t = dot(grad_output, b - a);
}

// Pow backward (element-wise)
// forward: out = pow(x, exp)
// backward: d_x = grad_out * exp * pow(x, exp - 1), d_exp = grad_out * out *
// log(x)
__device__ __forceinline__ void pow_bwd(float grad_output, float x, float exp, float output,
                                        float &grad_x, float &grad_exp) {
    grad_x = grad_output * exp * __powf(x, exp - 1.0f);
    grad_exp = grad_output * output * __logf(x + 1e-8f);
}

__device__ __forceinline__ void pow_bwd(const float3 &grad_output, const float3 &x, float exp,
                                        const float3 &output, float3 &grad_x, float &grad_exp) {
    grad_x.x = grad_output.x * exp * __powf(x.x, exp - 1.0f);
    grad_x.y = grad_output.y * exp * __powf(x.y, exp - 1.0f);
    grad_x.z = grad_output.z * exp * __powf(x.z, exp - 1.0f);

    grad_exp = dot(grad_output * output,
                   make_float3(__logf(x.x + 1e-8f), __logf(x.y + 1e-8f), __logf(x.z + 1e-8f)));
}

__device__ __forceinline__ void pow_bwd(const float3 &grad_output, const float3 &x,
                                        const float3 &exp, const float3 &output, float3 &grad_x,
                                        float3 &grad_exp) {
    grad_x.x = grad_output.x * exp.x * __powf(x.x, exp.x - 1.0f);
    grad_x.y = grad_output.y * exp.y * __powf(x.y, exp.y - 1.0f);
    grad_x.z = grad_output.z * exp.z * __powf(x.z, exp.z - 1.0f);

    grad_exp.x = grad_output.x * output.x * __logf(x.x + 1e-8f);
    grad_exp.y = grad_output.y * output.y * __logf(x.y + 1e-8f);
    grad_exp.z = grad_output.z * output.z * __logf(x.z + 1e-8f);
}

// ============================================================================
// Backward Pass Helpers for Matrix-Vector Operations
// ============================================================================

// NOTE: Old raw-array backward functions removed (mul_2x2_bwd, mul_3x3_bwd, mul_4x4_bwd).
// Use typed matrix functions above: mul_bwd(float2x2, ...) and mul_bwd(float3x3, ...)
// float4x4 operations are not used in the ISP pipeline.

// NOTE: Old raw-array matrix-matrix backward functions removed (mul_3x3_mat_bwd).
// Use typed matrix functions above: mul_mat_bwd(float2x2, ...) and mul_mat_bwd(float3x3, ...)

// ============================================================================
// Backward Pass Helpers for Scalar Functions
// ============================================================================

// exp backward: d_x = grad_out * exp(x) = grad_out * output
__device__ __forceinline__ void exp_bwd(float grad_output, float output, float &grad_x) {
    grad_x = grad_output * output;
}

// exp2 backward: d_x = grad_out * log(2) * exp2(x) = grad_out * log(2) *
// output
__device__ __forceinline__ void exp2_bwd(float grad_output, float output, float &grad_x) {
    grad_x = grad_output * 0.69314718f * output;  // log(2)
}

// log backward: d_x = grad_out / x
__device__ __forceinline__ void log_bwd(float grad_output, float x, float &grad_x) {
    grad_x = __fdividef(grad_output, x + 1e-8f);
}

// sqrt backward: d_x = grad_out / (2 * sqrt(x)) = grad_out / (2 * output)
__device__ __forceinline__ void sqrt_bwd(float grad_output, float output, float &grad_x) {
    grad_x = __fdividef(grad_output, 2.0f * output + 1e-8f);
}

// Division backward: out = a / b
// d_a = grad_out / b, d_b = -grad_out * a / (b * b)
__device__ __forceinline__ void div_bwd(float grad_output, float a, float b, float output,
                                        float &grad_a, float &grad_b) {
    grad_a = __fdividef(grad_output, b);
    grad_b = -grad_output * __fdividef(output, b);
}

// ============================================================================
// ISP-Specific Backward Helpers
// ============================================================================

// Softplus backward (for bounded positive parameters)
// forward: out = min_value + log(1 + exp(raw))
// backward: d_raw = grad_out * sigmoid(raw)
__device__ __forceinline__ void softplus_bwd(float grad_output, float raw, float &grad_raw) {
    float sigmoid = __fdividef(1.0f, 1.0f + __expf(-raw));
    grad_raw = grad_output * sigmoid;
}

// Sigmoid backward (for clamped parameters)
// forward: out = 1 / (1 + exp(-raw))
// backward: d_raw = grad_out * out * (1 - out)
__device__ __forceinline__ void sigmoid_bwd(float grad_output, float output, float &grad_raw) {
    grad_raw = grad_output * output * (1.0f - output);
}

// Vignetting falloff backward
// forward: falloff = 1 + alpha0*r2 + alpha1*r2^2 + alpha2*r2^3
// backward: d_r2 = grad_out * (alpha0 + 2*alpha1*r2 + 3*alpha2*r2^2)
//           d_alpha0 = grad_out * r2
//           d_alpha1 = grad_out * r2^2
//           d_alpha2 = grad_out * r2^3
__device__ __forceinline__ void vignetting_bwd(float grad_output, float r2, float alpha0,
                                               float alpha1, float alpha2, float &grad_r2,
                                               float &grad_alpha0, float &grad_alpha1,
                                               float &grad_alpha2) {
    float r4 = r2 * r2;
    float r6 = r4 * r2;

    grad_r2 = grad_output * __fmaf_rn(3.0f * alpha2, r4, __fmaf_rn(2.0f * alpha1, r2, alpha0));
    grad_alpha0 = grad_output * r2;
    grad_alpha1 = grad_output * r4;
    grad_alpha2 = grad_output * r6;
}

// Linear interpolation backward (for CRF)
// forward: out = y0 * (1 - w) + y1 * w
// backward: d_y0 = grad_out * (1 - w)
//           d_y1 = grad_out * w
//           d_w = grad_out * (y1 - y0)
__device__ __forceinline__ void lerp_interp_bwd(float grad_output, float y0, float y1, float w,
                                                float &grad_y0, float &grad_y1, float &grad_w) {
    grad_y0 = grad_output * (1.0f - w);
    grad_y1 = grad_output * w;
    grad_w = grad_output * (y1 - y0);
}

// ============================================================================
// Parameter Transformation Gradients
// ============================================================================

// Gradient of bounded_positive_forward: f(x) = min_val + log(1 + exp(x))
// This is a softplus function shifted by min_val
// Derivative: f'(x) = exp(x) / (1 + exp(x)) = sigmoid(x)
__device__ __forceinline__ float bounded_positive_backward(float raw, float min_value,
                                                           float grad_output) {
    float exp_x = __expf(raw);
    float sigmoid = __fdividef(exp_x, 1.0f + exp_x);
    return grad_output * sigmoid;
}

// Gradient of clamped_forward: f(x) = 1 / (1 + exp(-x)) = sigmoid(x)
// Derivative: f'(x) = sigmoid(x) * (1 - sigmoid(x))
__device__ __forceinline__ float clamped_backward(float raw, float grad_output) {
    float exp_neg_x = __expf(-raw);
    float sigmoid = __fdividef(1.0f, 1.0f + exp_neg_x);
    return grad_output * sigmoid * (1.0f - sigmoid);
}

// ============================================================================
// Modular ISP Component Backward Functions
// ============================================================================

// ----------------------------------------------------------------------------
// Exposure Compensation Backward
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_exposure_bwd(const float3 &rgb_in, float exposure_param,
                                                   const float3 &grad_rgb_out, float3 &grad_rgb_in,
                                                   float &grad_exposure_param) {
    // Forward: rgb_out = rgb_in * exp2(exposure_param)
    float exposure_factor = exp2f(exposure_param);

    // Gradient to exposure parameter
    // IMPORTANT: Compute this FIRST before modifying grad_rgb_in,
    // since grad_rgb_out and grad_rgb_in may alias the same memory!
    // d/d_exp[rgb_in * exp2(exp)] = rgb_in * exp2(exp) * ln(2) = rgb_out * ln(2)
    // grad_exposure = grad_rgb_out * (rgb_out * ln(2))
    float3 rgb_out = rgb_in * exposure_factor;
    grad_exposure_param = dot(grad_rgb_out, rgb_out) * 0.69314718f;  // ln(2)

    // Gradient to input
    grad_rgb_in = grad_rgb_out * exposure_factor;
}

// ----------------------------------------------------------------------------
// Vignetting Correction Backward
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_vignetting_bwd(
    const float3 &rgb_in, const VignettingChannelParams *vignetting_params,
    const float2 &pixel_coords, float resolution_x, float resolution_y, const float3 &grad_rgb_out,
    float3 &grad_rgb_in, VignettingChannelParams *grad_vignetting_params) {
    // Recompute forward
    float max_res = fmaxf(resolution_x, resolution_y);
    float2 uv = make_float2(__fdividef(pixel_coords.x - resolution_x * 0.5f, max_res),
                            __fdividef(pixel_coords.y - resolution_y * 0.5f, max_res));

    float rgb_arr[3] = {rgb_in.x, rgb_in.y, rgb_in.z};
    float grad_rgb_out_arr[3] = {grad_rgb_out.x, grad_rgb_out.y, grad_rgb_out.z};
    float grad_rgb_in_arr[3] = {0, 0, 0};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const VignettingChannelParams &params = vignetting_params[i];

        float dx = uv.x - params.cx;
        float dy = uv.y - params.cy;
        float r2 = __fmaf_rn(dx, dx, dy * dy);
        float r4 = r2 * r2;
        float r6 = r4 * r2;

        float falloff = __fmaf_rn(params.alpha2, r6,
                                  __fmaf_rn(params.alpha1, r4, __fmaf_rn(params.alpha0, r2, 1.0f)));
        float falloff_clamped = fmaxf(0.0f, fminf(1.0f, falloff));

        // Gradient to rgb_in
        grad_rgb_in_arr[i] = grad_rgb_out_arr[i] * falloff_clamped;

        // Gradient to falloff
        float grad_falloff = grad_rgb_out_arr[i] * rgb_arr[i];

        // Only compute gradients if not clamped
        // Use >= and <= to include boundary cases (falloff == 0.0 or falloff == 1.0)
        if (falloff >= 0.0f && falloff <= 1.0f) {
            // Gradient to alphas
            grad_vignetting_params[i].alpha0 += grad_falloff * r2;
            grad_vignetting_params[i].alpha1 += grad_falloff * r4;
            grad_vignetting_params[i].alpha2 += grad_falloff * r6;

            // Gradient to optical center through r2
            float grad_r2 =
                grad_falloff * __fmaf_rn(3.0f * params.alpha2, r4,
                                         __fmaf_rn(2.0f * params.alpha1, r2, params.alpha0));
            grad_vignetting_params[i].cx += -grad_r2 * 2.0f * dx;
            grad_vignetting_params[i].cy += -grad_r2 * 2.0f * dy;
        }
    }

    grad_rgb_in = make_float3(grad_rgb_in_arr[0], grad_rgb_in_arr[1], grad_rgb_in_arr[2]);
}

// ----------------------------------------------------------------------------
// Homography Backward - Helper Functions
// ----------------------------------------------------------------------------

// Backward for normalization: H_normalized = H / H[2][2]
// Given grad_H_normalized, compute grad_H_unnormalized
__device__ __forceinline__ void compute_homography_normalization_bwd(const float3x3 &H_unnorm,
                                                                     const float3x3 &grad_H_norm,
                                                                     float3x3 &grad_H_unnorm) {
    float s = H_unnorm(2, 2);

    if (fabsf(s) > 1.0e-20f) {
        float inv_s = 1.0f / s;
        float inv_s2 = inv_s * inv_s;

        // grad_H_unnorm[i] = grad_H_norm[i] * inv_s
        // grad_s = -sum(grad_H_norm[i] * H_unnorm[i] * inv_s^2)
        float grad_s = 0.0f;

#pragma unroll
        for (int i = 0; i < 9; i++) {
            grad_H_unnorm.m[i] = grad_H_norm.m[i] * inv_s;
            grad_s += -grad_H_norm.m[i] * H_unnorm.m[i] * inv_s2;
        }

        // Gradient flows back to H_unnorm[2][2]
        grad_H_unnorm(2, 2) += grad_s;
    } else {
        // No normalization occurred, gradient passes through unchanged
#pragma unroll
        for (int i = 0; i < 9; i++) {
            grad_H_unnorm.m[i] = grad_H_norm.m[i];
        }
    }
}

// Backward for diagonal matrix construction: D = diag(lambda)
// Given grad_D, compute grad_lambda
__device__ __forceinline__ void compute_homography_diagonal_matrix_bwd(const float3x3 &grad_D,
                                                                       float3 &grad_lambda) {
    grad_lambda.x = grad_D(0, 0);
    grad_lambda.y = grad_D(1, 1);
    grad_lambda.z = grad_D(2, 2);
}

// Backward for skew-symmetric matrix construction from t_gray
// skew = [[0, -t_gray.z, t_gray.y], [t_gray.z, 0, -t_gray.x], [-t_gray.y, t_gray.x, 0]]
__device__ __forceinline__ void compute_homography_skew_matrix_construction_bwd(
    const float3x3 &grad_skew, float3 &grad_t_gray) {
    grad_t_gray.x = 0.0f;
    grad_t_gray.y = 0.0f;
    grad_t_gray.z = 0.0f;

    // Skew matrix elements and their dependencies:
    // skew(0,1) = -t_gray.z
    grad_t_gray.z += -grad_skew(0, 1);
    // skew(0,2) = t_gray.y
    grad_t_gray.y += grad_skew(0, 2);
    // skew(1,0) = t_gray.z
    grad_t_gray.z += grad_skew(1, 0);
    // skew(1,2) = -t_gray.x
    grad_t_gray.x += -grad_skew(1, 2);
    // skew(2,0) = -t_gray.y
    grad_t_gray.y += -grad_skew(2, 0);
    // skew(2,1) = t_gray.x
    grad_t_gray.x += grad_skew(2, 1);
}

// Backward for T matrix construction from column vectors [t_b, t_r, t_g]
__device__ __forceinline__ void compute_homography_matrix_T_construction_bwd(const float3x3 &grad_T,
                                                                             float3 &grad_t_b,
                                                                             float3 &grad_t_r,
                                                                             float3 &grad_t_g) {
    // T is column-major: T(i,j) where j selects column (t_b=0, t_r=1, t_g=2)
    grad_t_b.x = grad_T(0, 0);
    grad_t_b.y = grad_T(1, 0);
    grad_t_b.z = grad_T(2, 0);

    grad_t_r.x = grad_T(0, 1);
    grad_t_r.y = grad_T(1, 1);
    grad_t_r.z = grad_T(2, 1);

    grad_t_g.x = grad_T(0, 2);
    grad_t_g.y = grad_T(1, 2);
    grad_t_g.z = grad_T(2, 2);
}

// Backward for target point construction: t = base + make_float3(offset.x, offset.y, 0)
__device__ __forceinline__ void compute_homography_target_point_bwd(const float3 &grad_t,
                                                                    float2 &grad_offset) {
    grad_offset.x = grad_t.x;
    grad_offset.y = grad_t.y;
    // grad_t.z doesn't contribute to offset (z is constant 1.0)
}

// Backward for nullspace computation via cross products with conditionals
// This handles the conditional logic in computing lambda
__device__ __forceinline__ void compute_homography_nullspace_computation_bwd(
    const float3x3 &M, const float3 &lambda_v, const float3x3 &grad_M_in, float3x3 &grad_M) {
    // Extract rows
    float3 r0 = make_float3(M(0, 0), M(0, 1), M(0, 2));
    float3 r1 = make_float3(M(1, 0), M(1, 1), M(1, 2));
    float3 r2 = make_float3(M(2, 0), M(2, 1), M(2, 2));

    // Recompute forward conditional path
    float3 lambda_test = cross(r0, r1);
    float n2 = dot(lambda_test, lambda_test);

    float3 grad_lambda = make_float3(0.0f, 0.0f, 0.0f);

    // Accumulate gradient to lambda from grad_M_in (which comes from diagonal matrix)
    // The gradient comes through the diagonal matrix D
    grad_lambda.x = grad_M_in(0, 0);
    grad_lambda.y = grad_M_in(1, 1);
    grad_lambda.z = grad_M_in(2, 2);

    float3 grad_r0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_r1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 grad_r2 = make_float3(0.0f, 0.0f, 0.0f);

    // Determine which branch was taken and backprop accordingly
    if (n2 < 1.0e-20f) {
        lambda_test = cross(r0, r2);
        n2 = dot(lambda_test, lambda_test);
        if (n2 < 1.0e-20f) {
            // lambda_v = cross(r1, r2)
            cross_bwd(grad_lambda, r1, r2, grad_r1, grad_r2);
        } else {
            // lambda_v = cross(r0, r2)
            cross_bwd(grad_lambda, r0, r2, grad_r0, grad_r2);
        }
    } else {
        // lambda_v = cross(r0, r1)
        cross_bwd(grad_lambda, r0, r1, grad_r0, grad_r1);
    }

    // Map row gradients back to matrix M
    grad_M = make_float3x3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    grad_M(0, 0) = grad_r0.x;
    grad_M(0, 1) = grad_r0.y;
    grad_M(0, 2) = grad_r0.z;

    grad_M(1, 0) = grad_r1.x;
    grad_M(1, 1) = grad_r1.y;
    grad_M(1, 2) = grad_r1.z;

    grad_M(2, 0) = grad_r2.x;
    grad_M(2, 1) = grad_r2.y;
    grad_M(2, 2) = grad_r2.z;
}

// ----------------------------------------------------------------------------
// Homography Backward - Main Function
// ----------------------------------------------------------------------------
/**
 * @brief Backward pass for compute_homography
 *
 * This function computes the gradient of color_params given the gradient of H.
 * The forward pass is decomposed into subfunctions which are independently differentiated.
 */
__device__ __forceinline__ void compute_homography_bwd(const ColorPPISPParams *color_params,
                                                       const float3x3 &grad_H,
                                                       ColorPPISPParams *grad_color_params) {
    // ========================================================================
    // Step 1: Recompute forward pass to get all intermediates
    // ========================================================================

    const float2 &b_lat = color_params->b;
    const float2 &r_lat = color_params->r;
    const float2 &g_lat = color_params->g;
    const float2 &n_lat = color_params->n;

    // Map latent to real offsets via ZCA 2x2 blocks
    float2x2 zca_b, zca_r, zca_g, zca_n;
    zca_b.m[0] = COLOR_PINV_BLOCKS[0][0];
    zca_b.m[1] = COLOR_PINV_BLOCKS[0][1];
    zca_b.m[2] = COLOR_PINV_BLOCKS[0][2];
    zca_b.m[3] = COLOR_PINV_BLOCKS[0][3];

    zca_r.m[0] = COLOR_PINV_BLOCKS[1][0];
    zca_r.m[1] = COLOR_PINV_BLOCKS[1][1];
    zca_r.m[2] = COLOR_PINV_BLOCKS[1][2];
    zca_r.m[3] = COLOR_PINV_BLOCKS[1][3];

    zca_g.m[0] = COLOR_PINV_BLOCKS[2][0];
    zca_g.m[1] = COLOR_PINV_BLOCKS[2][1];
    zca_g.m[2] = COLOR_PINV_BLOCKS[2][2];
    zca_g.m[3] = COLOR_PINV_BLOCKS[2][3];

    zca_n.m[0] = COLOR_PINV_BLOCKS[3][0];
    zca_n.m[1] = COLOR_PINV_BLOCKS[3][1];
    zca_n.m[2] = COLOR_PINV_BLOCKS[3][2];
    zca_n.m[3] = COLOR_PINV_BLOCKS[3][3];

    float2 bd = zca_b * b_lat;
    float2 rd = zca_r * r_lat;
    float2 gd = zca_g * g_lat;
    float2 nd = zca_n * n_lat;

    float3 t_b = make_float3(0.0f + bd.x, 0.0f + bd.y, 1.0f);
    float3 t_r = make_float3(1.0f + rd.x, 0.0f + rd.y, 1.0f);
    float3 t_g = make_float3(0.0f + gd.x, 1.0f + gd.y, 1.0f);
    float3 t_gray = make_float3(1.0f / 3.0f + nd.x, 1.0f / 3.0f + nd.y, 1.0f);

    float3x3 T = make_float3x3(t_b.x, t_r.x, t_g.x, t_b.y, t_r.y, t_g.y, t_b.z, t_r.z, t_g.z);

    float3x3 skew = make_float3x3(0.0f, -t_gray.z, t_gray.y, t_gray.z, 0.0f, -t_gray.x, -t_gray.y,
                                  t_gray.x, 0.0f);

    float3x3 M = skew * T;

    float3 r0 = make_float3(M(0, 0), M(0, 1), M(0, 2));
    float3 r1 = make_float3(M(1, 0), M(1, 1), M(1, 2));
    float3 r2 = make_float3(M(2, 0), M(2, 1), M(2, 2));

    float3 lambda_v = cross(r0, r1);
    float n2 = dot(lambda_v, lambda_v);

    if (n2 < 1.0e-20f) {
        lambda_v = cross(r0, r2);
        n2 = dot(lambda_v, lambda_v);
        if (n2 < 1.0e-20f) {
            lambda_v = cross(r1, r2);
        }
    }

    float3x3 S_inv = make_float3x3(-1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

    float3x3 D =
        make_float3x3(lambda_v.x, 0.0f, 0.0f, 0.0f, lambda_v.y, 0.0f, 0.0f, 0.0f, lambda_v.z);

    float3x3 TD = T * D;
    float3x3 H_unnorm = TD * S_inv;

    // ========================================================================
    // Step 2: Backward pass through operations in reverse order
    // ========================================================================

    // Initialize gradient accumulators
    float3x3 grad_H_unnorm = make_float3x3(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    // 2.1: Backprop through normalization
    compute_homography_normalization_bwd(H_unnorm, grad_H, grad_H_unnorm);

    // 2.2: Backprop through H = TD * S_inv
    // grad_TD = grad_H_unnorm * S_inv^T (since S_inv is constant)
    float3x3 grad_TD, grad_S_inv_unused;
    mul_mat_bwd(TD, S_inv, grad_H_unnorm, grad_TD, grad_S_inv_unused);

    // 2.3: Backprop through TD = T * D
    float3x3 grad_T, grad_D;
    mul_mat_bwd(T, D, grad_TD, grad_T, grad_D);

    // 2.4: Backprop through D = diag(lambda)
    float3 grad_lambda;
    compute_homography_diagonal_matrix_bwd(grad_D, grad_lambda);

    // 2.5: Backprop through lambda computation (nullspace via cross products)
    // Pass grad_lambda through a wrapper that looks like grad_D for the interface
    float3x3 grad_D_for_nullspace = make_float3x3(grad_lambda.x, 0.0f, 0.0f, 0.0f, grad_lambda.y,
                                                  0.0f, 0.0f, 0.0f, grad_lambda.z);
    float3x3 grad_M;
    compute_homography_nullspace_computation_bwd(M, lambda_v, grad_D_for_nullspace, grad_M);

    // 2.6: Backprop through M = skew * T
    float3x3 grad_skew, grad_T_from_M;
    mul_mat_bwd(skew, T, grad_M, grad_skew, grad_T_from_M);

    // Accumulate gradient to T
#pragma unroll
    for (int i = 0; i < 9; i++) {
        grad_T.m[i] += grad_T_from_M.m[i];
    }

    // 2.7: Backprop through skew matrix construction
    float3 grad_t_gray;
    compute_homography_skew_matrix_construction_bwd(grad_skew, grad_t_gray);

    // 2.8: Backprop through T matrix construction
    float3 grad_t_b, grad_t_r, grad_t_g;
    compute_homography_matrix_T_construction_bwd(grad_T, grad_t_b, grad_t_r, grad_t_g);

    // 2.9: Backprop through target point construction
    float2 grad_bd, grad_rd, grad_gd, grad_nd;
    compute_homography_target_point_bwd(grad_t_b, grad_bd);
    compute_homography_target_point_bwd(grad_t_r, grad_rd);
    compute_homography_target_point_bwd(grad_t_g, grad_gd);
    compute_homography_target_point_bwd(grad_t_gray, grad_nd);

    // 2.10: Backprop through ZCA transforms: offset = zca * latent
    float2x2 dummy_grad_zca;
    float2 grad_b_lat, grad_r_lat, grad_g_lat, grad_n_lat;

    mul_bwd(zca_b, b_lat, grad_bd, dummy_grad_zca, grad_b_lat);
    mul_bwd(zca_r, r_lat, grad_rd, dummy_grad_zca, grad_r_lat);
    mul_bwd(zca_g, g_lat, grad_gd, dummy_grad_zca, grad_g_lat);
    mul_bwd(zca_n, n_lat, grad_nd, dummy_grad_zca, grad_n_lat);

    // 2.11: Accumulate gradients to output
    grad_color_params->b.x += grad_b_lat.x;
    grad_color_params->b.y += grad_b_lat.y;
    grad_color_params->r.x += grad_r_lat.x;
    grad_color_params->r.y += grad_r_lat.y;
    grad_color_params->g.x += grad_g_lat.x;
    grad_color_params->g.y += grad_g_lat.y;
    grad_color_params->n.x += grad_n_lat.x;
    grad_color_params->n.y += grad_n_lat.y;
}

// ----------------------------------------------------------------------------
// Color Correction - PPISP Backward
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_color_correction_ppisp_bwd(
    const float3 &rgb_in, const ColorPPISPParams *color_params, const float3 &grad_rgb_out,
    float3 &grad_rgb_in, ColorPPISPParams *grad_color_params) {
    // Recompute forward
    float3x3 H = compute_homography(color_params);

    float intensity = rgb_in.x + rgb_in.y + rgb_in.z;
    float3 rgi_in = make_float3(rgb_in.x, rgb_in.y, intensity);

    float3 rgi_out = H * rgi_in;

    float norm_factor = __fdividef(intensity, rgi_out.z + 1.0e-5f);
    float3 rgi_out_norm = rgi_out * norm_factor;

    // Backward: rgb = [rgi[0], rgi[1], rgi[2] - rgi[0] - rgi[1]]
    float3 grad_rgi_out_norm;
    grad_rgi_out_norm.x = grad_rgb_out.x - grad_rgb_out.z;
    grad_rgi_out_norm.y = grad_rgb_out.y - grad_rgb_out.z;
    grad_rgi_out_norm.z = grad_rgb_out.z;

    // Gradient through normalization
    float3 grad_rgi_out = grad_rgi_out_norm * norm_factor;

    // Gradient to norm_factor through rgi_out
    float grad_norm_factor = dot(grad_rgi_out_norm, rgi_out);

    // Gradient to rgi_out.z through norm_factor
    float grad_rgi_out_z_norm = -grad_norm_factor * norm_factor / (rgi_out.z + 1.0e-5f);
    grad_rgi_out.z += grad_rgi_out_z_norm;

    // Gradient through homography: rgi_out = H * rgi_in
    // Use typed matrix backward: grad_rgi_in = H^T * grad_rgi_out
    float3x3 grad_H;
    float3 grad_rgi_in;
    mul_bwd(H, rgi_in, grad_rgi_out, grad_H, grad_rgi_in);

    // Gradient through RGI construction: rgi_in = [r, g, r+g+b]
    grad_rgb_in.x = grad_rgi_in.x + grad_rgi_in.z;
    grad_rgb_in.y = grad_rgi_in.y + grad_rgi_in.z;
    grad_rgb_in.z = grad_rgi_in.z;

    // Gradient through intensity (intensity = rgb_in.x + rgb_in.y + rgb_in.z)
    float grad_intensity = 0.0f;
    if (intensity > 1e-8f) {
        grad_intensity = grad_norm_factor * norm_factor / intensity;
    }

    // Distribute gradient to all RGB channels (since intensity = sum of all)
    grad_rgb_in.x += grad_intensity;
    grad_rgb_in.y += grad_intensity;
    grad_rgb_in.z += grad_intensity;

    // Backprop through compute_homography to get gradients for color_params
    compute_homography_bwd(color_params, grad_H, grad_color_params);
}

// ----------------------------------------------------------------------------
// CRF - PPISP Backward
// ----------------------------------------------------------------------------
__device__ __forceinline__ void apply_crf_ppisp_bwd(const float3 &rgb_in,
                                                    const CRFPPISPChannelParams *crf_params,
                                                    const float3 &grad_rgb_out, float3 &grad_rgb_in,
                                                    CRFPPISPChannelParams *grad_crf_params) {
    float3 rgb_clamped = clamp(rgb_in, 0.0f, 1.0f);
    float rgb_arr[3] = {rgb_clamped.x, rgb_clamped.y, rgb_clamped.z};
    float grad_out_arr[3] = {grad_rgb_out.x, grad_rgb_out.y, grad_rgb_out.z};
    float grad_in_arr[3] = {0, 0, 0};

#pragma unroll
    for (int i = 0; i < 3; i++) {
        const CRFPPISPChannelParams &params = crf_params[i];

        // Recompute forward
        float toe = bounded_positive_forward(params.toe, 0.3f);
        float shoulder = bounded_positive_forward(params.shoulder, 0.3f);
        float gamma = bounded_positive_forward(params.gamma, 0.1f);
        float center = clamped_forward(params.center);

        float lerp_val = __fmaf_rn(shoulder - toe, center, toe);
        float a = __fdividef(shoulder * center, lerp_val);
        float b = 1.0f - a;

        float x = rgb_arr[i];
        float y;

        if (x <= center) {
            y = a * __powf(__fdividef(x, center), toe);
        } else {
            y = 1.0f - b * __powf(__fdividef(1.0f - x, 1.0f - center), shoulder);
        }

        float output = __powf(fmaxf(0.0f, y), gamma);
        float y_clamped = fmaxf(0.0f, y);

        // Backward through gamma
        float grad_y = 0.0f;
        if (y_clamped > 0.0f) {
            grad_y = grad_out_arr[i] * gamma * __powf(y_clamped, gamma - 1.0f);
        }

        // Backward through piecewise curve
        float grad_x = 0.0f;
        if (x <= center && center > 0.0f) {
            float base = __fdividef(x, center);
            if (base > 0.0f) {
                grad_x = grad_y * a * toe * __powf(base, toe - 1.0f) / center;
            }
        } else if (x > center && center < 1.0f) {
            float base = __fdividef(1.0f - x, 1.0f - center);
            if (base > 0.0f) {
                grad_x = grad_y * b * shoulder * __powf(base, shoulder - 1.0f) / (1.0f - center);
            }
        }

        grad_in_arr[i] = grad_x;

        // Parameter gradients through transformations
        // We need to compute: grad_raw_param = grad_transformed_param *
        // d_transformed/d_raw

        // First, compute gradients to the transformed parameters (toe, shoulder,
        // gamma, center)
        float grad_toe = 0.0f;
        float grad_shoulder = 0.0f;
        float grad_gamma = 0.0f;
        float grad_center = 0.0f;

        // Gradient to gamma from output = pow(y_clamped, gamma)
        if (y_clamped > 0.0f) {
            grad_gamma = grad_out_arr[i] * output * __logf(y_clamped + 1e-8f);
        }

        // Gradients to toe, shoulder, center through the piecewise curve
        // These are complex, so we'll compute them numerically stable

        // For toe and shoulder, they affect the curve through a, b, and the power
        // terms For center, it affects both the conditional and the curve shape

        // Gradient to 'a' and 'b' from the curve
        float grad_a = 0.0f;
        float grad_b = 0.0f;

        if (x <= center && center > 0.0f) {
            // y = a * (x/center)^toe
            float base = __fdividef(x, center);
            if (base > 0.0f) {
                float powered = __powf(base, toe);
                grad_a += grad_y * powered;

                // grad_toe from the power term
                grad_toe += grad_y * a * powered * __logf(base + 1e-8f);

                // grad_center from the base (x/center)
                // dy/dcenter = dy/dbase * dbase/dcenter
                // dy/dbase = grad_y * a * toe * base^(toe-1)
                // dbase/dcenter = d(x/center)/dcenter = -x / center^2
                float grad_base = grad_y * a * toe * __powf(base, toe - 1.0f);
                grad_center += grad_base * (-x / (center * center));
            }
        } else if (x > center && center < 1.0f) {
            // y = 1 - b * ((1-x)/(1-center))^shoulder
            float base = __fdividef(1.0f - x, 1.0f - center);
            if (base > 0.0f) {
                float powered = __powf(base, shoulder);
                grad_b += -grad_y * powered;

                // grad_shoulder from the power term
                grad_shoulder += -grad_y * b * powered * __logf(base + 1e-8f);

                // grad_center from the base ((1-x)/(1-center))
                // dy/dcenter = dy/dbase * dbase/dcenter
                // dy/dbase = grad_y * (-b * shoulder * base^(shoulder-1))
                // dbase/dcenter = d((1-x)/(1-center))/dcenter = (1-x) / (1-center)^2
                float grad_base = grad_y * (-b * shoulder * __powf(base, shoulder - 1.0f));
                float dbase_dcenter = (1.0f - x) / ((1.0f - center) * (1.0f - center));
                grad_center += grad_base * dbase_dcenter;
            }
        }

        // Gradient to toe, shoulder through a and b
        // a = (shoulder * center) / lerp_val, where lerp_val = (shoulder - toe) *
        // center + toe b = 1 - a

        // NOTE: In Slang, the computation is structured as:
        //   1. Vectorially compute: float3 a = (shoulders * centers) / lerp(toes, shoulders,
        //   centers)
        //   2. Per-channel loop: float c = centers[i]; ... use c in divisions
        // The key is that 'centers' is used in BOTH places, but Slang's autodiff
        // only computes gradients through the VECTORIAL use (#1), not through
        // the per-channel extraction (#2).
        //
        // In CUDA, we're computing everything per-channel, so we need to ensure
        // gradients ONLY flow through the 'a' and 'b' computation, NOT through
        // the per-channel divisions (x/c, (1-x)/(1-c)) which we already removed.

        // CRITICAL: Since b = 1 - a, we have db/da = -1
        // So grad_a needs to accumulate -grad_b BEFORE we use it
        grad_a += -grad_b;

        float grad_lerp_val = 0.0f;
        if (fabsf(lerp_val) > 1e-8f) {
            // grad_a (now including contribution from grad_b) contributes to shoulder,
            // center, and lerp_val
            float a_over_lerp = __fdividef(shoulder * center, lerp_val);
            grad_shoulder += grad_a * center / lerp_val;
            grad_center += grad_a * shoulder / lerp_val;
            grad_lerp_val += -grad_a * a_over_lerp / lerp_val;
        }

        // Gradient through lerp_val = (shoulder - toe) * center + toe
        grad_shoulder += grad_lerp_val * center;
        grad_toe += grad_lerp_val * (1.0f - center);
        grad_center += grad_lerp_val * (shoulder - toe);

        // Now transform gradients back through the parameter transformations
        grad_crf_params[i].toe += bounded_positive_backward(params.toe, 0.3f, grad_toe);
        grad_crf_params[i].shoulder +=
            bounded_positive_backward(params.shoulder, 0.3f, grad_shoulder);
        grad_crf_params[i].gamma += bounded_positive_backward(params.gamma, 0.1f, grad_gamma);
        grad_crf_params[i].center += clamped_backward(params.center, grad_center);
    }

    grad_rgb_in = make_float3(grad_in_arr[0], grad_in_arr[1], grad_in_arr[2]);
}

```



### File: setup.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Setup script for ppisp CUDA extension.
Based on https://github.com/rahul-goel/fused-ssim/blob/main/setup.py
"""

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import torch
import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stderr.reconfigure(line_buffering=True)


def get_cuda_arch_flags():
    """
    Determine CUDA architecture flags with the following priority:
    1. TORCH_CUDA_ARCH_LIST environment variable (standard PyTorch convention)
    2. Runtime GPU detection via torch.cuda.get_device_capability()
    3. Conservative fallback defaults (no Blackwell/compute_120, requires CUDA 12.8+)
    """
    # Priority 1: Check TORCH_CUDA_ARCH_LIST environment variable
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list_env:
        # Parse format: "7.5;8.0;8.6" or "7.5 8.0 8.6"
        arch_entries = arch_list_env.replace(";", " ").split()
        gencode_flags = []
        for entry in arch_entries:
            entry = entry.strip()
            if not entry:
                continue
            # Handle entries like "8.6" or "8.6+PTX"
            has_ptx = "+PTX" in entry
            arch_version = entry.replace("+PTX", "").strip()
            if "." in arch_version:
                major, minor = arch_version.split(".")
                compute_arch = f"compute_{major}{minor}"
                sm_arch = f"sm_{major}{minor}"
            else:
                # Handle entries like "86"
                compute_arch = f"compute_{arch_version}"
                sm_arch = f"sm_{arch_version}"
            if has_ptx:
                gencode_flags.append(f"-gencode=arch={compute_arch},code={compute_arch}")
            gencode_flags.append(f"-gencode=arch={compute_arch},code={sm_arch}")
        if gencode_flags:
            msg = f"Using TORCH_CUDA_ARCH_LIST: {arch_list_env}"
            print(msg)
            print(msg, file=sys.stderr, flush=True)
            return gencode_flags, f"TORCH_CUDA_ARCH_LIST={arch_list_env}"

    # Priority 2: Runtime GPU detection
    try:
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            compute_capability = torch.cuda.get_device_capability(device)
            arch = f"sm_{compute_capability[0]}{compute_capability[1]}"
            msg = f"Detected GPU architecture: {arch}"
            print(msg)
            print(msg, file=sys.stderr, flush=True)
            return [f"-arch={arch}"], arch
    except Exception as e:
        error_msg = f"Failed to detect GPU architecture: {e}"
        print(error_msg)
        print(error_msg, file=sys.stderr, flush=True)

    # Priority 3: Conservative fallback defaults
    # Note: compute_120 (Blackwell) requires CUDA 12.8+, so it's excluded
    fallback_archs = [
        "-gencode=arch=compute_75,code=sm_75",   # Turing, CUDA 10+
        "-gencode=arch=compute_80,code=sm_80",   # Ampere, CUDA 11+
        "-gencode=arch=compute_86,code=sm_86",   # Ampere GA10x, CUDA 11.1+
        "-gencode=arch=compute_89,code=sm_89",   # Ada Lovelace, CUDA 11.8+
        "-gencode=arch=compute_90,code=sm_90",   # Hopper, CUDA 12.0+
    ]
    msg = "Using conservative fallback architectures (no Blackwell/compute_120)"
    print(msg)
    print(msg, file=sys.stderr, flush=True)
    return fallback_archs, "multiple architectures (fallback)"


# Get CUDA architecture flags
arch_flags, detected_arch = get_cuda_arch_flags()

nvcc_args = [
    "-O3",
    "--use_fast_math",
]
nvcc_args += ["-lineinfo", "--generate-line-info", "--source-in-ptx"]
nvcc_args.extend(arch_flags)


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        arch_info = f"Building with GPU architecture: {detected_arch if detected_arch else 'multiple architectures'}"
        print("\n" + "=" * 50)
        print(arch_info)
        print("=" * 50 + "\n")
        super().build_extensions()


setup(
    ext_modules=[
        CUDAExtension(
            name="ppisp_cuda",
            sources=[
                "ppisp/src/ppisp_impl.cu",
                "ppisp/ext.cpp"
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": nvcc_args
            }
        )
    ],
    cmdclass={
        'build_ext': CustomBuildExtension
    }
)

final_msg = f"Setup completed. NVCC args: {nvcc_args}"
print(final_msg)

```



### File: tests\README.md

```markdown
# PPISP Tests

This directory contains tests for the PPISP CUDA implementation.

## Test Files

- **test_cuda_vs_torch.py**: Core correctness tests
  - Compares CUDA implementation against pure PyTorch reference
  - Forward pass equivalence (multiple configurations)
  - Backward pass gradient equivalence (all parameter groups)
  - Tests for disabled effects (camera_idx=-1, frame_idx=-1)
  - Stress tests (large batch, many cameras/frames)

- **test_gradcheck.py**: Numerical gradient verification
  - Uses `torch.autograd.gradcheck` for independent gradient validation
  - Gradient magnitude sanity checks

- **torch_reference.py**: PyTorch reference implementation (not tests)
  - Pure PyTorch implementation of PPISP pipeline
  - Used as ground truth for correctness testing

## Running Tests

### Run all tests:
```bash
pytest tests/ -v
```

### Run specific test file:
```bash
pytest tests/test_cuda_vs_torch.py -v
```

### Run specific test:
```bash
pytest tests/test_cuda_vs_torch.py::test_forward_basic -v
```

## Requirements

- PyTorch with CUDA support
- pytest
- CUDA-capable GPU

## Test Coverage

- Forward pass correctness (CUDA vs PyTorch reference)
- Backward pass correctness (all gradient outputs)
- Numerical gradient verification via finite differences
- Different batch sizes (1 to 4096 pixels)
- Multiple cameras and frames configurations
- Disabled effects (camera_idx=-1, frame_idx=-1)
- Gradient magnitude sanity checks

```



### File: tests\__init__.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

```



### File: tests\test_cuda_vs_torch.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test to validate CUDA backend numerical equivalence with Torch backend.

This test ensures that the CUDA implementation produces identical results
to the pure PyTorch reference implementation for both forward and backward passes.
"""

import torch
import pytest
import numpy as np
from typing import Dict, Tuple

import ppisp_cuda
import ppisp
from tests.torch_reference import ppisp_apply_torch


# =============================================================================
# Test Utilities
# =============================================================================

def create_test_params(
    num_cameras: int = 2,
    num_frames: int = 5,
    seed: int = 42,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create PPISP parameters with random values.

    Parameters are kept small to ensure numerical stability.
    """
    torch.manual_seed(seed)

    # Exposure: [num_frames] - small range to keep output in valid range
    exposure_params = torch.empty(
        num_frames, device=device, dtype=dtype
    ).uniform_(-0.5, 0.5).requires_grad_(True)

    # Vignetting: [num_cameras, 3, 5] - small polynomial coefficients
    vignetting_params = torch.empty(
        num_cameras, 3, 5, device=device, dtype=dtype
    ).uniform_(-0.1, 0.1).requires_grad_(True)

    # Color correction: [num_frames, 8] - small latent offsets
    color_params = torch.empty(
        num_frames, 8, device=device, dtype=dtype
    ).uniform_(-0.1, 0.1).requires_grad_(True)

    # CRF parameters: [num_cameras, 3, 4] - small raw parameter values
    crf_params = torch.empty(
        num_cameras, 3, 4, device=device, dtype=dtype
    ).uniform_(-0.5, 0.5).requires_grad_(True)

    return {
        'exposure_params': exposure_params,
        'vignetting_params': vignetting_params,
        'color_params': color_params,
        'crf_params': crf_params,
    }


def create_test_inputs(
    batch_size: int = 256,
    resolution_w: int = 512,
    resolution_h: int = 512,
    num_cameras: int = 2,
    num_frames: int = 5,
    seed: int = 42,
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Create consistent test inputs.

    RGB values are kept in [0.1, 0.9] range to avoid edge cases in CRF.
    Pixel coordinates avoid edges for stable vignetting gradients.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Random RGB values in [0.1, 0.9] range (avoid edge cases)
    rgb = torch.empty(batch_size, 3, device=device,
                      dtype=dtype).uniform_(0.1, 0.9)

    # Pixel coordinates - avoid edges for stable vignetting
    pixel_coords = torch.empty(batch_size, 2, device=device, dtype=dtype)
    pixel_coords[:, 0].uniform_(0.1 * resolution_w, 0.9 * resolution_w)
    pixel_coords[:, 1].uniform_(0.1 * resolution_h, 0.9 * resolution_h)

    # Random camera and frame indices
    camera_idx = torch.randint(0, num_cameras, (1,)).item()
    frame_idx = torch.randint(0, num_frames, (1,)).item()

    return {
        'rgb': rgb,
        'pixel_coords': pixel_coords,
        'resolution_w': resolution_w,
        'resolution_h': resolution_h,
        'camera_idx': camera_idx,
        'frame_idx': frame_idx,
    }


class CUDAAutograd(torch.autograd.Function):
    """Autograd wrapper for CUDA PPISP."""

    @staticmethod
    def forward(ctx, exposure_params, vignetting_params,
                color_params, crf_params, rgb_in, pixel_coords,
                resolution_w, resolution_h, camera_idx, frame_idx):

        # Ensure contiguous float32
        exposure_params = exposure_params.float().contiguous()
        vignetting_params = vignetting_params.float().contiguous()
        color_params = color_params.float().contiguous()
        crf_params = crf_params.float().contiguous()
        rgb_in = rgb_in.float().contiguous()
        pixel_coords = pixel_coords.float().contiguous()

        rgb_out = ppisp_cuda.ppisp_forward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, pixel_coords,
            resolution_w, resolution_h, camera_idx, frame_idx,
        )

        ctx.save_for_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, rgb_out, pixel_coords
        )
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h
        ctx.camera_idx = camera_idx
        ctx.frame_idx = frame_idx

        return rgb_out

    @staticmethod
    def backward(ctx, v_rgb_out):
        (exposure_params, vignetting_params,
         color_params, crf_params, rgb_in, rgb_out, pixel_coords) = ctx.saved_tensors

        grads = ppisp_cuda.ppisp_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, rgb_out, pixel_coords,
            v_rgb_out.contiguous(),
            ctx.resolution_w, ctx.resolution_h,
            ctx.camera_idx, ctx.frame_idx,
        )

        return grads + (None,) * 5  # None for pixel_coords + 4 int args


def run_cuda_forward(params, inputs, rgb):
    """Run CUDA forward pass."""
    return CUDAAutograd.apply(
        params['exposure_params'],
        params['vignetting_params'],
        params['color_params'],
        params['crf_params'],
        rgb,
        inputs['pixel_coords'],
        inputs['resolution_w'],
        inputs['resolution_h'],
        inputs['camera_idx'],
        inputs['frame_idx'],
    )


def run_torch_forward(params, inputs, rgb):
    """Run PyTorch forward pass."""
    return ppisp_apply_torch(
        params['exposure_params'],
        params['vignetting_params'],
        params['color_params'],
        params['crf_params'],
        rgb,
        inputs['pixel_coords'],
        inputs['resolution_w'],
        inputs['resolution_h'],
        inputs['camera_idx'],
        inputs['frame_idx'],
    )


def compute_relative_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Compute relative error between two tensors."""
    abs_diff = torch.abs(a - b)
    abs_max = torch.maximum(torch.abs(a), torch.abs(b))
    rel_error = abs_diff / (abs_max + eps)
    return rel_error.max().item()


# =============================================================================
# Forward Pass Tests
# =============================================================================

def test_forward_basic():
    """Test basic forward pass equivalence."""
    params = create_test_params(num_cameras=2, num_frames=5, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=2, num_frames=5, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Forward pass max diff: {max_diff}"


def test_forward_multiple_iterations():
    """Test forward pass with multiple random parameter configurations."""
    num_iterations = 10
    atol = 1e-5  # Relaxed tolerance for float32 precision

    for iteration in range(num_iterations):
        params = create_test_params(
            num_cameras=2, num_frames=5, seed=42 + iteration)
        inputs = create_test_inputs(
            batch_size=256, num_cameras=2, num_frames=5, seed=100 + iteration)

        rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
        rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

        max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
        assert max_diff <= atol, f"Iteration {iteration}: max_diff={max_diff}"


def test_forward_different_batch_sizes():
    """Test forward pass with different batch sizes."""
    for batch_size in [1, 16, 128, 512, 1024]:
        params = create_test_params(seed=42)
        inputs = create_test_inputs(batch_size=batch_size, seed=42)

        rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
        rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

        max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
        assert max_diff < 3e-6, f"Batch size {batch_size}: max_diff={max_diff}"


def test_forward_identity_params():
    """Test forward pass with identity parameters (zeros)."""
    params = create_test_params(seed=42)

    # Set all params to zero (identity transform)
    with torch.no_grad():
        params['exposure_params'].zero_()
        params['vignetting_params'].zero_()
        params['color_params'].zero_()
        params['crf_params'].zero_()

    inputs = create_test_inputs(batch_size=256, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Identity params max_diff={max_diff}"


def test_forward_no_camera_effects():
    """Test forward pass with camera_idx=-1 (disabled camera effects)."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['camera_idx'] = -1

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"No camera effects max_diff={max_diff}"


def test_forward_no_frame_effects():
    """Test forward pass with frame_idx=-1 (disabled frame effects)."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['frame_idx'] = -1

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"No frame effects max_diff={max_diff}"


# =============================================================================
# Backward Pass Tests
# =============================================================================

def test_backward_basic():
    """Test backward pass gradient equivalence between CUDA and PyTorch.

    Compares all gradient outputs (rgb_in, exposure, vignetting, color, crf)
    between CUDA backward pass and PyTorch autograd reference.
    """
    params_cuda = create_test_params(num_cameras=2, num_frames=5, seed=42)
    params_torch = create_test_params(num_cameras=2, num_frames=5, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=2, num_frames=5, seed=42)

    # Clone inputs for both passes
    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    # Forward passes
    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    # Identical gradient for backward
    grad_output = torch.randn_like(output_cuda)

    # Backward passes
    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # Check each gradient with detailed error info
    grad_pairs = [
        ('rgb_in', rgb_cuda.grad, rgb_torch.grad),
        ('exposure', params_cuda['exposure_params'].grad,
         params_torch['exposure_params'].grad),
        ('vignetting', params_cuda['vignetting_params'].grad,
         params_torch['vignetting_params'].grad),
        ('color', params_cuda['color_params'].grad,
         params_torch['color_params'].grad),
        ('crf', params_cuda['crf_params'].grad,
         params_torch['crf_params'].grad),
    ]

    for name, grad_cuda, grad_torch in grad_pairs:
        max_diff = torch.abs(grad_cuda - grad_torch).max().item()
        rel_error = compute_relative_error(grad_cuda, grad_torch)
        assert max_diff <= atol, \
            f"{name} grad: max_diff={max_diff:.2e}, rel_error={rel_error:.2e}"


def test_backward_no_camera_effects():
    """Test backward pass with camera_idx=-1 (disabled camera effects)."""
    params_cuda = create_test_params(seed=42)
    params_torch = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['camera_idx'] = -1

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    grad_output = torch.randn_like(output_cuda)

    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # RGB and per-frame params should have gradients
    max_diff = torch.abs(rgb_cuda.grad - rgb_torch.grad).max().item()
    assert max_diff <= atol, f"RGB grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['exposure_params'].grad -
        params_torch['exposure_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Exposure grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['color_params'].grad -
        params_torch['color_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Color grad max_diff={max_diff}"


def test_backward_no_frame_effects():
    """Test backward pass with frame_idx=-1."""
    params_cuda = create_test_params(seed=42)
    params_torch = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=256, seed=42)
    inputs['frame_idx'] = -1

    rgb_cuda = inputs['rgb'].clone().requires_grad_(True)
    rgb_torch = inputs['rgb'].clone().requires_grad_(True)

    output_cuda = run_cuda_forward(params_cuda, inputs, rgb_cuda)
    output_torch = run_torch_forward(params_torch, inputs, rgb_torch)

    grad_output = torch.randn_like(output_cuda)

    output_cuda.backward(grad_output)
    output_torch.backward(grad_output)

    atol = 1e-4

    # RGB and per-camera params should have gradients
    max_diff = torch.abs(rgb_cuda.grad - rgb_torch.grad).max().item()
    assert max_diff <= atol, f"RGB grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['vignetting_params'].grad -
        params_torch['vignetting_params'].grad
    ).max().item()
    assert max_diff <= atol, f"Vignetting grad max_diff={max_diff}"

    max_diff = torch.abs(
        params_cuda['crf_params'].grad -
        params_torch['crf_params'].grad
    ).max().item()
    assert max_diff <= atol, f"CRF grad max_diff={max_diff}"


# =============================================================================
# Stress Tests
# =============================================================================

def test_large_batch():
    """Test with large batch size."""
    params = create_test_params(seed=42)
    inputs = create_test_inputs(batch_size=4096, seed=42)

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Large batch max_diff={max_diff}"


def test_many_cameras_frames():
    """Test with many cameras and frames."""
    num_cameras = 10
    num_frames = 50

    params = create_test_params(
        num_cameras=num_cameras, num_frames=num_frames, seed=42)
    inputs = create_test_inputs(
        batch_size=256, num_cameras=num_cameras, num_frames=num_frames, seed=42
    )

    rgb_cuda = run_cuda_forward(params, inputs, inputs['rgb'].clone())
    rgb_torch = run_torch_forward(params, inputs, inputs['rgb'].clone())

    max_diff = torch.abs(rgb_cuda - rgb_torch).max().item()
    assert max_diff < 3e-6, f"Many cameras/frames max_diff={max_diff}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

```



### File: tests\test_gradcheck.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient checking tests using torch.autograd.gradcheck.

Validates CUDA gradient implementations against numerical finite differences.
These tests complement the CUDA-vs-PyTorch comparison tests by verifying
gradients through an independent numerical method.

Note: The CUDA kernel uses float32 internally, so gradcheck tolerances are
set accordingly. Tests use float64 for numerical differentiation precision
but actual kernel computations are in float32.
"""

import torch
import pytest
from torch.autograd import gradcheck

import ppisp_cuda


# Tolerances for float32 precision
GRADCHECK_EPS = 1e-3     # Perturbation for finite differences
GRADCHECK_ATOL = 5e-3    # Absolute tolerance
GRADCHECK_RTOL = 5e-2    # Relative tolerance (5%)


class PPISPAutograd(torch.autograd.Function):
    """Autograd wrapper for CUDA PPISP forward/backward."""

    @staticmethod
    def forward(
        ctx,
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    ):
        # Ensure contiguous float32
        exposure_params = exposure_params.float().contiguous()
        vignetting_params = vignetting_params.float().contiguous()
        color_params = color_params.float().contiguous()
        crf_params = crf_params.float().contiguous()
        rgb_in = rgb_in.float().contiguous()
        pixel_coords = pixel_coords.float().contiguous()

        rgb_out = ppisp_cuda.ppisp_forward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            pixel_coords,
            resolution_w,
            resolution_h,
            camera_idx,
            frame_idx,
        )

        ctx.save_for_backward(
            exposure_params, vignetting_params,
            color_params, crf_params, rgb_in, rgb_out, pixel_coords
        )
        ctx.resolution_w = resolution_w
        ctx.resolution_h = resolution_h
        ctx.camera_idx = camera_idx
        ctx.frame_idx = frame_idx

        return rgb_out

    @staticmethod
    def backward(ctx, v_rgb_out):
        (exposure_params, vignetting_params,
         color_params, crf_params, rgb_in, rgb_out, pixel_coords) = ctx.saved_tensors

        (v_exposure_params, v_vignetting_params,
         v_color_params, v_crf_params, v_rgb_in) = ppisp_cuda.ppisp_backward(
            exposure_params,
            vignetting_params,
            color_params,
            crf_params,
            rgb_in,
            rgb_out,
            pixel_coords,
            v_rgb_out.contiguous(),
            ctx.resolution_w,
            ctx.resolution_h,
            ctx.camera_idx,
            ctx.frame_idx,
        )

        return (
            v_exposure_params,
            v_vignetting_params,
            v_color_params,
            v_crf_params,
            v_rgb_in,
            None,  # pixel_coords
            None,  # resolution_w
            None,  # resolution_h
            None,  # camera_idx
            None,  # frame_idx
        )


def create_test_inputs(
    num_pixels=5,
    num_cameras=1,
    num_frames=1,
    resolution_w=64,
    resolution_h=64,
    camera_idx=0,
    frame_idx=0,
    requires_grad=True,
    device='cuda',
    dtype=torch.float64,
    seed=42,
):
    """Create test inputs for gradient checking.

    Uses small parameter perturbations around identity to avoid numerical issues.
    Note: CUDA kernel internally uses float32, so dtype=float64 only affects
    gradcheck's numerical differentiation precision.
    """
    torch.manual_seed(seed)

    exposure_params = torch.randn(
        num_frames, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.1

    vignetting_params = torch.randn(
        num_cameras, 3, 5, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.01

    # Color params: very small perturbations to keep homography near identity
    color_params = torch.randn(
        num_frames, 8, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.02

    # CRF params: initialized near identity
    crf_params = torch.randn(
        num_cameras, 3, 4, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.1

    # RGB input in valid range [0.2, 0.8] to avoid edge cases in CRF
    rgb_in = torch.rand(
        num_pixels, 3, device=device, dtype=dtype, requires_grad=requires_grad
    ) * 0.5 + 0.25

    # Pixel coordinates - avoid extreme edges for vignetting
    pixel_coords = torch.rand(
        num_pixels, 2, device=device, dtype=dtype, requires_grad=False
    )
    pixel_coords[:, 0] = pixel_coords[:, 0] * 0.6 + 0.2  # 20%-80% of width
    pixel_coords[:, 1] = pixel_coords[:, 1] * 0.6 + 0.2  # 20%-80% of height
    pixel_coords[:, 0] *= resolution_w
    pixel_coords[:, 1] *= resolution_h

    return {
        'exposure_params': exposure_params,
        'vignetting_params': vignetting_params,
        'color_params': color_params,
        'crf_params': crf_params,
        'rgb_in': rgb_in,
        'pixel_coords': pixel_coords,
        'resolution_w': resolution_w,
        'resolution_h': resolution_h,
        'camera_idx': camera_idx,
        'frame_idx': frame_idx,
    }


def ppisp_wrapper(
    exposure_params,
    vignetting_params,
    color_params,
    crf_params,
    rgb_in,
    pixel_coords,
    resolution_w,
    resolution_h,
    camera_idx,
    frame_idx,
):
    """Wrapper for gradcheck that handles double->float conversion."""
    return PPISPAutograd.apply(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        camera_idx,
        frame_idx,
    )


def test_gradcheck_all_params():
    """Test gradients for all parameters simultaneously using finite differences.

    This is the primary numerical gradient verification test. It validates that
    the CUDA backward pass produces gradients consistent with numerical
    differentiation for all parameter groups at once.
    """
    inputs = create_test_inputs(num_pixels=3)

    def func(exp, vig, col, crf, rgb):
        return ppisp_wrapper(
            exp, vig, col, crf, rgb,
            inputs['pixel_coords'],
            inputs['resolution_w'],
            inputs['resolution_h'],
            inputs['camera_idx'],
            inputs['frame_idx'],
        )

    assert gradcheck(
        func,
        (
            inputs['exposure_params'],
            inputs['vignetting_params'],
            inputs['color_params'],
            inputs['crf_params'],
            inputs['rgb_in'],
        ),
        eps=GRADCHECK_EPS,
        atol=GRADCHECK_ATOL,
        rtol=GRADCHECK_RTOL,
        raise_exception=True,
    )


def test_gradient_magnitude_reasonable():
    """Verify gradient magnitudes are reasonable (not exploding/vanishing).

    This sanity check ensures gradients are finite and within expected bounds,
    catching issues like NaN propagation or gradient explosion.
    """
    torch.manual_seed(42)

    num_pixels = 20
    resolution_w, resolution_h = 64, 64

    # Create leaf tensors
    exposure_params = torch.empty(
        1, device='cuda', dtype=torch.float32
    ).normal_(0, 0.1).requires_grad_(True)

    vignetting_params = torch.empty(
        1, 3, 5, device='cuda', dtype=torch.float32
    ).normal_(0, 0.01).requires_grad_(True)

    color_params = torch.empty(
        1, 8, device='cuda', dtype=torch.float32
    ).normal_(0, 0.02).requires_grad_(True)

    crf_params = torch.empty(
        1, 3, 4, device='cuda', dtype=torch.float32
    ).normal_(0, 0.1).requires_grad_(True)

    rgb_in = torch.empty(
        num_pixels, 3, device='cuda', dtype=torch.float32
    ).uniform_(0.25, 0.75).requires_grad_(True)

    pixel_coords = torch.empty(
        num_pixels, 2, device='cuda', dtype=torch.float32
    ).uniform_(0, 1)
    pixel_coords[:, 0] = pixel_coords[:, 0] * resolution_w
    pixel_coords[:, 1] = pixel_coords[:, 1] * resolution_h

    rgb_out = ppisp_wrapper(
        exposure_params,
        vignetting_params,
        color_params,
        crf_params,
        rgb_in,
        pixel_coords,
        resolution_w,
        resolution_h,
        0,  # camera_idx
        0,  # frame_idx
    )

    # Backward with unit gradient
    rgb_out.sum().backward()

    # Check gradients exist and are finite
    for name, param in [
        ('exposure_params', exposure_params),
        ('vignetting_params', vignetting_params),
        ('color_params', color_params),
        ('crf_params', crf_params),
        ('rgb_in', rgb_in),
    ]:
        assert param.grad is not None, f"{name} has no gradient"
        assert torch.isfinite(param.grad).all(
        ), f"{name} has non-finite gradient"

        grad_norm = param.grad.norm().item()
        assert grad_norm < 1e6, f"{name} gradient too large: {grad_norm}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

```



### File: tests\torch_reference.py

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
PyTorch reference implementation of PPISP pipeline.

This is a minimal, unoptimized port of the CUDA kernel for testing purposes:
- Verifying CUDA kernel outputs against a readable reference
- Gradient verification via torch.autograd
- Prototyping changes before porting to CUDA

The CUDA implementation is the authoritative source of truth for production.
"""

import torch
import torch.nn.functional as F

# ZCA pinv blocks for color correction [Blue, Red, Green, Neutral]
# Stored as 8x8 block-diagonal matrix for efficient single-matmul application
_COLOR_PINV_BLOCK_DIAG = torch.block_diag(
    torch.tensor([[0.0480542, -0.0043631], [-0.0043631, 0.0481283]]),  # Blue
    torch.tensor([[0.0580570, -0.0179872], [-0.0179872, 0.0431061]]),  # Red
    torch.tensor([[0.0433336, -0.0180537], [-0.0180537, 0.0580500]]),  # Green
    torch.tensor([[0.0128369, -0.0034654],
                 [-0.0034654, 0.0128158]]),  # Neutral
).to(torch.float32)


def _get_homography_torch(color_params: torch.Tensor, frame_idx: int) -> torch.Tensor:
    """Compute color correction homography matrix from latent params.

    Minimal PyTorch port of get_homography() from ppisp.slang.
    """
    cp = color_params[frame_idx]  # [8]

    # Map latent -> real offsets via ZCA block-diagonal matrix
    block_diag = _COLOR_PINV_BLOCK_DIAG.to(cp.device)
    offsets = cp @ block_diag  # [8]

    # Extract real RG offsets for each control chromaticity
    bd = offsets[0:2]  # blue
    rd = offsets[2:4]  # red
    gd = offsets[4:6]  # green
    nd = offsets[6:8]  # neutral/gray

    # Fixed source chromaticities (r, g, 1)
    s_b = torch.tensor([0.0, 0.0, 1.0], device=cp.device)
    s_r = torch.tensor([1.0, 0.0, 1.0], device=cp.device)
    s_g = torch.tensor([0.0, 1.0, 1.0], device=cp.device)
    s_gray = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0], device=cp.device)

    # Target = source + offset
    t_b = torch.stack([s_b[0] + bd[0], s_b[1] + bd[1], torch.ones_like(bd[0])])
    t_r = torch.stack([s_r[0] + rd[0], s_r[1] + rd[1], torch.ones_like(rd[0])])
    t_g = torch.stack([s_g[0] + gd[0], s_g[1] + gd[1], torch.ones_like(gd[0])])
    t_gray = torch.stack(
        [s_gray[0] + nd[0], s_gray[1] + nd[1], torch.ones_like(nd[0])])

    # T = [t_b, t_r, t_g] as columns
    T = torch.stack([t_b, t_r, t_g], dim=1)  # [3, 3]

    # Skew-symmetric matrix of t_gray
    skew = torch.stack([
        torch.stack([torch.zeros_like(t_gray[0]), -t_gray[2], t_gray[1]]),
        torch.stack([t_gray[2], torch.zeros_like(t_gray[0]), -t_gray[0]]),
        torch.stack([-t_gray[1], t_gray[0], torch.zeros_like(t_gray[0])]),
    ])  # [3, 3]

    # M = skew @ T
    M = skew @ T

    # Nullspace vector lambda via cross of two independent rows.
    # Compute all three cross products and select the one with largest magnitude
    # to avoid data-dependent control flow (breaks torch.compile).
    r0, r1, r2 = M[0], M[1], M[2]
    lam01 = torch.linalg.cross(r0, r1)
    lam02 = torch.linalg.cross(r0, r2)
    lam12 = torch.linalg.cross(r1, r2)

    n01 = (lam01 * lam01).sum()
    n02 = (lam02 * lam02).sum()
    n12 = (lam12 * lam12).sum()

    # Select cross product with largest magnitude (no branching)
    lam = torch.where(n01 >= n02,
                      torch.where(n01 >= n12, lam01, lam12),
                      torch.where(n02 >= n12, lam02, lam12))

    # Precomputed inverse of S = [s_b s_r s_g]
    S_inv = torch.tensor([
        [-1.0, -1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], device=cp.device)

    # H = T @ diag(lambda) @ S_inv
    D = torch.diag(lam)
    H = T @ D @ S_inv

    # Normalize so H[2,2] ~ 1 (using eps to avoid branching; intensity is
    # renormalized after applying H anyway)
    H = H / (H[2, 2] + 1e-10)

    return H


def ppisp_apply_torch(
    exposure_params: torch.Tensor,
    vignetting_params: torch.Tensor,
    color_params: torch.Tensor,
    crf_params: torch.Tensor,
    rgb_in: torch.Tensor,
    pixel_coords: torch.Tensor,
    resolution_w: int,
    resolution_h: int,
    camera_idx: int,
    frame_idx: int,
) -> torch.Tensor:
    """PyTorch reference implementation of PPISP pipeline.

    This is a minimal, unoptimized port of the CUDA kernel for testing.
    Use the CUDA kernel for production.

    Args:
        exposure_params: Per-frame exposure [num_frames]
        vignetting_params: Per-camera vignetting [num_cameras, 3, 5]
        color_params: Per-frame color correction [num_frames, 8]
        crf_params: Per-camera CRF [num_cameras, 3, 4]
        rgb_in: Input RGB [N, 3]
        pixel_coords: Pixel coordinates [N, 2]
        resolution_w: Image width
        resolution_h: Image height
        camera_idx: Camera index (-1 to disable per-camera effects)
        frame_idx: Frame index (-1 to disable per-frame effects)

    Returns:
        Processed RGB [N, 3]
    """
    rgb = rgb_in.clone()

    # --- Exposure compensation ---
    if frame_idx != -1:
        rgb = rgb * \
            torch.pow(torch.tensor(2.0, device=rgb.device),
                      exposure_params[frame_idx])

    # --- Vignetting ---
    def compute_vignetting_falloff(vig_params: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
        """Compute vignetting falloff from params [5] and UV coords [N, 2]."""
        optical_center = vig_params[:2]
        alphas = vig_params[2:]
        delta = uv - optical_center.unsqueeze(0)
        r2 = (delta * delta).sum(dim=-1)
        falloff = torch.ones_like(r2)
        r2_pow = r2.clone()
        for alpha in alphas:
            falloff = falloff + alpha * r2_pow
            r2_pow = r2_pow * r2
        return falloff.clamp(0.0, 1.0)

    if camera_idx != -1:
        res_f = torch.tensor([resolution_w, resolution_h],
                             device=rgb.device, dtype=rgb.dtype)
        uv = (pixel_coords - res_f * 0.5) / res_f.max()

        # Per-channel vignetting [num_cameras, 3, 5]
        rgb_channels = []
        for ch in range(3):
            falloff = compute_vignetting_falloff(
                vignetting_params[camera_idx, ch], uv)
            rgb_channels.append(rgb[:, ch] * falloff)

        rgb = torch.stack(rgb_channels, dim=-1)

    # --- Color correction ---
    if frame_idx != -1:
        # Homography-based color correction
        H = _get_homography_torch(color_params, frame_idx)  # [3, 3]

        # RGB -> RGI (Red, Green, Intensity)
        intensity = rgb.sum(dim=-1, keepdim=True)  # [N, 1]
        rgi = torch.cat([rgb[:, 0:1], rgb[:, 1:2], intensity],
                        dim=-1)  # [N, 3]

        # Apply homography
        rgi = (H @ rgi.T).T  # [N, 3]

        # Renormalize intensity
        rgi = rgi * (intensity / (rgi[:, 2:3] + 1e-5))

        # RGI -> RGB
        r_out = rgi[:, 0]
        g_out = rgi[:, 1]
        b_out = rgi[:, 2] - r_out - g_out
        rgb = torch.stack([r_out, g_out, b_out], dim=-1)

    # --- CRF (Camera Response Function) ---
    if camera_idx != -1:
        # Parametric toe-shoulder CRF
        rgb = rgb.clamp(0.0, 1.0)

        rgb_channels = []
        for ch in range(3):
            # [4]: toe_raw, shoulder_raw, gamma_raw, center_raw
            crf = crf_params[camera_idx, ch]

            # Decode parameters with bounded activations
            toe = 0.3 + F.softplus(crf[0])       # min 0.3
            shoulder = 0.3 + F.softplus(crf[1])  # min 0.3
            gamma = 0.1 + F.softplus(crf[2])     # min 0.1
            center = torch.sigmoid(crf[3])       # 0-1

            # Compute a, b for piecewise curve
            lerp_val = toe + center * (shoulder - toe)
            a = (shoulder * center) / lerp_val
            b = 1.0 - a

            x = rgb[:, ch]
            mask_low = x <= center

            # Use eps clamp to avoid NaN gradients from pow(0, fractional)
            eps = 1e-6
            y_low = a * torch.pow((x / center).clamp(min=eps), toe)
            y_high = 1.0 - b * \
                torch.pow(((1.0 - x) / (1.0 - center)
                           ).clamp(min=eps), shoulder)

            # Select based on mask
            y = torch.where(mask_low, y_low, y_high)

            # Apply gamma
            rgb_channels.append(torch.pow(y.clamp(min=eps), gamma))

        rgb = torch.stack(rgb_channels, dim=-1)

    return rgb

```

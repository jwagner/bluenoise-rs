use std::num::NonZeroU64;

use image::GrayImage;
use rand::{seq::SliceRandom, Rng, SeedableRng};

use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "bluenoise.png")]
    output: String,

    #[arg(short, long, default_value = "64")]
    size: u16,

    #[arg(long, default_value = "1.5")]
    sigma: f32,

    #[arg(long)]
    seed: Option<NonZeroU64>,

    #[arg(long)]
    debug: Option<String>,
}

fn main() {
    let args = Args::parse();
    let mut rng = if let Some(seed) = args.seed {
        rand_chacha::ChaCha8Rng::seed_from_u64(seed.into())
    } else {
        rand_chacha::ChaCha8Rng::from_entropy()
    };
    let mut generator = BlueNoiseGenerator::new(args.size as i32, args.sigma);
    generator.generate(&mut rng);

    generator.save_dither_array(&args.output);

    if let Some(debug) = args.debug {
        generator.save_debug(&debug);
    }
}

#[derive(Debug, Default)]
struct BlueNoiseGenerator {
    size: i32, // always positive
    sigma: f32,
    binary_pattern: Vec<u8>,
    // could use symmetry to quarter it's size
    filter_lut: Vec<f32>,
    density: Vec<f32>,
    dither_array: Vec<i32>,
    ones: i32, // always positive,
}

impl BlueNoiseGenerator {
    fn new(size: i32, sigma: f32) -> Self {
        let n = (size * size) as usize;
        Self {
            size,
            sigma,
            binary_pattern: vec![0; n],
            filter_lut: vec![0.; n],
            density: vec![0.; n],
            dither_array: vec![0; n],
            ones: 0,
        }
    }

    #[inline]
    // changes a value in the binary pattern
    // updates ones and density
    // o(size^2)
    fn set_binary_pattern(&mut self, x: i32, y: i32, value: u8) {
        debug_assert!(value == 0 || value == 1);
        let s = self.size;
        let x = x.rem_euclid(s);
        let y = y.rem_euclid(s);
        let i = (x + y * s) as usize;
        let old = self.binary_pattern[i];
        self.binary_pattern[i] = value;
        let delta = value as i32 - old as i32;
        self.ones += delta;
        debug_assert!(self.ones >= 0);
        self.update_density(x, y, delta as f32);
    }

    #[inline]
    fn update_density(&mut self, x: i32, y: i32, delta: f32) {
        let s = self.size;
        for v in 0..s {
            for u in 0..s {
                let fx = (s / 2 + (u - x)).rem_euclid(s);
                let fy = (s / 2 + (v - y)).rem_euclid(s);
                let fi = (fx + fy * s) as usize;
                let i = (u + v * s) as usize;
                self.density[i] += self.filter_lut[fi] * delta;
            }
        }
    }

    fn generate(&mut self, rng: &mut impl Rng) {
        self.initialize_filter_lut();
        self.generate_prototype_pattern(rng);
        self.generate_initial_binary_pattern();
        self.generate_dither_array();
    }

    fn initialize_filter_lut(&mut self) {
        let s = self.size;
        let inv_denominator = 1. / (2. * self.sigma * self.sigma);
        for v in 0..s {
            for u in 0..s {
                let i = (v + u * s) as usize;
                let x = (u - s / 2) as f32;
                let y = (v - s / 2) as f32;
                let r_squared = x * x + y * y;
                self.filter_lut[i] = (-r_squared * inv_denominator).exp();
            }
        }
    }

    fn generate_prototype_pattern(&mut self, rng: &mut impl Rng) {
        let mut ranks: Vec<i32> = Vec::from_iter(0..self.size * self.size);
        ranks.shuffle(rng);
        let ones = self.size * self.size / 2 - 1;
        // let ones = 3;
        for y in 0..self.size {
            for x in 0..self.size {
                let i = (x + y * self.size) as usize;
                if ranks[i] < ones {
                    self.set_binary_pattern(x, y, 1);
                }
            }
        }
        // self.ones = ones;
    }

    fn generate_initial_binary_pattern(&mut self) {
        for _ in 0..1_000_000 {
            let (cx, cy) = self.find_tightest_cluster();
            self.set_binary_pattern(cx, cy, 0);
            let (vx, vy) = self.find_largest_void();
            self.set_binary_pattern(vx, vy, 1);
            if vx == cx && vy == cy {
                return;
            }
        }
        panic!("iteration limit hit");
    }

    fn find_tightest_cluster(&self) -> (i32, i32) {
        // argmax
        let i = self
            .density
            .iter()
            .enumerate()
            .filter(|(i, _)| self.binary_pattern[*i] == 1)
            .max_by(|(_, &a), (_, b)| a.total_cmp(b))
            .expect("density not to be empty")
            .0 as i32;
        let x = i % self.size;
        let y = i / self.size;
        (x, y)
    }

    fn find_largest_void(&self) -> (i32, i32) {
        // argmin
        let i = self
            .density
            .iter()
            .enumerate()
            .filter(|(i, _)| self.binary_pattern[*i] == 0)
            .min_by(|(_, &a), (_, b)| a.total_cmp(b))
            .expect("density not to be empty")
            .0 as i32;
        let x = i % self.size;
        let y = i / self.size;
        (x, y)
    }

    fn generate_dither_array(&mut self) {
        self.generate_dither_array_phase_one();
        self.generate_dither_array_phase_two();
    }

    fn save_debug(&self, prefix: &str) {
        let density_scale = 1.0
            / self
                .density
                .iter()
                .max_by(|&&a, &b| a.total_cmp(b))
                .expect("density not to be empty");
        let density_values: Vec<u8> = self
            .density
            .iter()
            .map(|x| (x * density_scale * 255.) as u8)
            .collect();

        let s = self.size as u32;

        let density_img = GrayImage::from_vec(s, s, density_values).expect("image");
        density_img
            .save(prefix.to_owned() + "_density.png")
            .expect("save to succeed");

        let binary_pattern_values: Vec<u8> = self.binary_pattern.iter().map(|x| x * 255).collect();
        GrayImage::from_vec(s, s, binary_pattern_values)
            .unwrap()
            .save(prefix.to_owned() + "_bp.png")
            .expect("save to succeed");

        self.save_dither_array(&(prefix.to_owned() + "_da.png"));
    }

    fn save_dither_array(&self, name: &str) {
        let s = self.size as u32;
        let dither_array_scale = 255.0
            / (*self
                .dither_array
                .iter()
                .max()
                .expect("dither array not to be empty")) as f32;
        let dither_array_values: Vec<u8> = self
            .dither_array
            .iter()
            .map(|x| ((*x as f32) * dither_array_scale) as u8)
            .collect();
        GrayImage::from_vec(s, s, dither_array_values)
            .unwrap()
            .save(name)
            .expect("save to succeed");
    }

    fn generate_dither_array_phase_one(&mut self) {
        let mut rank = self.ones - 1;
        while rank >= 0 {
            let (x, y) = self.find_tightest_cluster();
            self.set_binary_pattern(x, y, 0);
            self.set_dither_array(x, y, rank);
            rank -= 1;
        }
    }

    #[inline]
    fn set_dither_array(&mut self, x: i32, y: i32, rank: i32) {
        let i = (x + y * self.size) as usize;
        self.dither_array[i] = rank;
    }

    fn generate_dither_array_phase_two(&mut self) {
        let mut rank = self.ones;
        let half = (self.size * self.size) / 2;
        while rank < half * 2 {
            let (x, y) = self.find_largest_void();
            self.set_binary_pattern(x, y, 1);
            self.set_dither_array(x, y, rank);
            rank += 1;
        }
    }
}

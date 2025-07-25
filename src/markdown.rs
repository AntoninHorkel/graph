#![feature(stdarch_x86_avx512)]

#[cfg(target_arch = "x86")]
use std::arch::x86::*; // TODO: popcnt and cvtmask
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    __m128i,
    __m256i,
    __m512i,
    _cvtmask64_u64,
    _mm_and_si128,
    _mm_loadu_si128,
    _mm_movemask_epi8,
    _mm_shuffle_epi8,
    _mm_srli_epi16,
    _mm256_and_si256,
    _mm256_loadu_si256,
    _mm256_movemask_epi8,
    _mm256_shuffle_epi8,
    _mm256_srli_epi16,
    _mm512_mask_compressstoreu_epi8,
    _mm512_shuffle_epi8,
    _mm512_srli_epi16,
    _mm512_test_epi8_mask,
    _popcnt64,
};
use std::mem::transmute;

const fn tables<const N: usize>(bytes: &[u8]) -> ([u8; N], [u8; N]) {
    assert!(N >= 16);
    let mut low_table = [0_u8; N];
    let mut high_table = [0_u8; N];
    let mut i = 0;
    while i < bytes.len() {
        low_table[(bytes[i] & 0x0F) as usize] |= 1 << (bytes[i] >> 4);
        i += 1;
    }
    let mut i = 0;
    while i < 16 {
        high_table[i] = if i < 8 { 1 << i } else { 0xFF };
        i += 1;
    }
    let mut i = 16;
    while i < N {
        low_table[i] = low_table[i % 16];
        high_table[i] = high_table[i % 16];
        i += 1;
    }
    (low_table, high_table)
}

macro_rules! tables {
    (ssse3, $bytes:expr) => {
        tables!(16, $bytes)
    };
    (avx2, $bytes:expr) => {
        tables!(32, $bytes)
    };
    (avx512, $bytes:expr) => {
        tables!(64, $bytes)
    };
    ($n:expr, $bytes:expr) => {
        const {
            let (low_table, high_table) = tables::<$n>($bytes);
            #[allow(unsafe_code)]
            unsafe {
                (transmute(low_table), transmute(high_table))
            }
        }
    };
}

const fn range<const N: usize>(start: u8, step: u8) -> [u8; N] {
    let mut table = [start; N];
    let mut i = 1;
    while i < N {
        table[i] = table[i - 1] + step;
        i += 1;
    }
    table
}

macro_rules! range {
    (avx512, $start:expr, $step:expr) => {
        range!(64, $start, $step)
    };
    ($n:expr, $start:expr, $step:expr) => {
        const {
            let table = range::<$n>($start, $step);
            #[allow(unsafe_code)]
            unsafe {
                transmute(table)
            }
        }
    };
}

#[doc = "https://validark.dev/posts/eine-kleine-vectorized-classification/"]
#[inline]
#[target_feature(enable = "sse2,ssse3")]
fn ssse3(chunk: &[u8], tokens: &mut [u8], indices: &mut [u8]) -> usize {
    #[allow(unsafe_code)]
    let chunk = unsafe { _mm_loadu_si128(chunk.as_ptr() as _) };
    let mut mask = {
        let (low_table, high_table) = tables!(ssse3, b"[]()`\n");
        let low_shuffled = _mm_shuffle_epi8(low_table, chunk);
        let high_shuffled = _mm_shuffle_epi8(high_table, _mm_srli_epi16(chunk, 4));
        _mm_movemask_epi8(_mm_and_si128(low_shuffled, high_shuffled))
    };
    // std::hint::unlikely?
    // while mask != 0 {
    //     _popcnt32
    //     _blsr_u32
    // }
    todo!()
}

#[doc = "https://validark.dev/posts/eine-kleine-vectorized-classification/"]
#[inline]
#[target_feature(enable = "avx,avx2")]
fn avx2(chunk: &[u8], tokens: &mut [u8], indices: &mut [u8]) -> usize {
    #[allow(unsafe_code)]
    let chunk = unsafe { _mm256_loadu_si256(chunk.as_ptr() as _) };
    let mut mask = {
        let (low_table, high_table) = tables!(avx2, b"[]()`\n");
        let low_shuffled = _mm256_shuffle_epi8(low_table, chunk);
        let high_shuffled = _mm256_shuffle_epi8(high_table, _mm256_srli_epi16(chunk, 4));
        _mm256_movemask_epi8(_mm256_and_si256(low_shuffled, high_shuffled))
    };
    // std::hint::unlikely?
    // while mask != 0 {
    //     _popcnt32
    //     _blsr_u32
    // }
    todo!()
}

#[doc = "https://validark.dev/posts/eine-kleine-vectorized-classification/"]
#[inline]
#[target_feature(enable = "avx512bw,popcnt")]
fn avx512bw(chunk: &[u8], tokens: &mut [u8], indices: &mut [u8]) -> usize {
    #[allow(unsafe_code)]
    let chunk = unsafe { _mm512_loadu_epi8(chunk.as_ptr() as _) };
    let mask = {
        let (low_table, high_table) = tables!(avx512, b"[]()`\n");
        let low_shuffled = _mm512_shuffle_epi8(low_table, chunk);
        let high_shuffled = _mm512_shuffle_epi8(high_table, _mm512_srli_epi16(chunk, 4));
        _mm512_test_epi8_mask(low_shuffled, high_shuffled)
    };
    todo!();
    _popcnt64(_cvtmask64_u64(mask) as _) as _
}

#[doc = "https://validark.dev/posts/eine-kleine-vectorized-classification/"]
#[inline]
#[target_feature(enable = "avx512bw,avx512vbmi,popcnt")]
fn avx512vbmi(chunk: &[u8], tokens: &mut [u8], indices: &mut [u8]) -> usize {
    #[allow(unsafe_code)]
    let chunk = unsafe { _mm512_loadu_epi8(chunk.as_ptr() as _) };
    let mask = {
        let (low_table, high_table) = tables!(avx512, b"[]()`\n");
        let low_shuffled = _mm512_shuffle_epi8(low_table, chunk);
        let high_shuffled = _mm512_shuffle_epi8(high_table, _mm512_srli_epi16(chunk, 4));
        _mm512_test_epi8_mask(low_shuffled, high_shuffled)
    };
    let range = range!(avx512, 0, 1); // _mm512_set_epi8
    #[allow(unsafe_code)]
    unsafe {
        _mm512_mask_compressstoreu_epi8(tokens.as_mut_ptr() as _, mask, chunk);
        _mm512_mask_compressstoreu_epi8(indices.as_mut_ptr() as _, mask, range);
    }
    _popcnt64(_cvtmask64_u64(mask) as _) as _
}

#[doc = "https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512"]
#[inline]
fn avx512vp2intersect(chunk: __m512i, tokens: &mut [u8], indices: &mut [u8]) -> usize {
    // _mm512_2intersect_epi64()
    // _mm512_ternarylogic_epi64()
    todo!("As if anyone actually has a CPU that supports AVX-512 VP2INTERSECT")
}

#[inline]
#[target_feature(enable = "avx512bw")]
fn parse(bytes: &[u8]) {
    let mut chunks = bytes.chunks_exact(64);
    chunks.for_each(|chunk| {});
    let chunk = chunks.remainder();
}

fn rename_me(chunk: &[u8], tokens: &mut [u8], indices: &mut [u8]) -> usize {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("popcnt") {
            if is_x86_feature_detected!("avx512bw") {
                if is_x86_feature_detected!("avx512vbmi") {
                    #[allow(unsafe_code)]
                    unsafe {
                        avx512vbmi(chunk, tokens, indices)
                    }
                } else {
                    #[allow(unsafe_code)]
                    unsafe {
                        avx512bw(chunk, tokens, indices)
                    }
                }
            } else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("avx2") {
                #[allow(unsafe_code)]
                unsafe {
                    avx2(chunk, tokens, indices)
                }
            } else if is_x86_feature_detected!("sse2") && is_x86_feature_detected!("ssse3") {
                #[allow(unsafe_code)]
                unsafe {
                    ssse3(chunk, tokens, indices)
                }
            } else {
                todo!()
            }
        } else {
            todo!()
        }
    }
}

// let high_lookup = _mm_shuffle_epi8(
//     high_table,
//     _mm_and_si128(
//         _mm_srli_epi16(chunk, 4),
//         const {
//             #[allow(unsafe_code)]
//             unsafe {
//                 transmute([0x0F_u8; 16])
//             }
//         },
//     ),
// );

enum Classification {
    EineKleine,
    Kor,
    Vp2intersect,
}

#[inline]
#[target_feature(enable = "avx512f,avx512bw,avx512vbmi2")]
fn test<const CLASSIFICATION: Classification>(
    chunk: u8x64,
    tokens: &mut MaybeUninit<[u8; 64]>,
    indices: &mut MaybeUninit<[u8; 64]>,
) {
    let mask = match CLASSIFICATION {
        Classification::EineKleine => {
            // https://validark.dev/posts/eine-kleine-vectorized-classification/
            let (low_table, high_table) = const { build_tables(b"[]()`\n") };
            let low_lookup = _mm512_shuffle_epi8(low_table, chunk);
            let high_nibbles = _mm512_and_si512(_mm512_srli_epi16(chunk, 4), const { splat(0x0F) });
            let high_bits = _mm512_shuffle_epi8(high_table, high_nibbles);
            _mm512_test_epi8_mask(low_lookup, high_bits)
        }
        Classification::Kor => _kor_mask64(
            _kor_mask64(
                _kor_mask64(
                    _mm512_cmpeq_epi8_mask(chunk, const { splat(b'[') }),
                    _mm512_cmpeq_epi8_mask(chunk, const { splat(b']') }),
                ),
                _kor_mask64(
                    _mm512_cmpeq_epi8_mask(chunk, const { splat(b'(') }),
                    _mm512_cmpeq_epi8_mask(chunk, const { splat(b')') }),
                ),
            ),
            _kor_mask64(
                _mm512_cmpeq_epi8_mask(chunk, const { splat(b'`') }),
                _mm512_cmpeq_epi8_mask(chunk, const { splat(b'\n') }),
            ),
        ),
        Classification::Vp2intersect => {
            // _mm512_2intersect_epi64()
            // _mm512_ternarylogic_epi64()
            todo!(
                "As if anyone actually has a CPU that supports AVX-512 VP2INTERSECT (https://en.wikipedia.org/wiki/AVX-512#CPUs_with_AVX-512)"
            )
        }
    };
    let a = const { range(0, 1) };
    #[allow(unsafe_code)]
    unsafe {
        _mm512_mask_compressstoreu_epi8(tokens.as_mut_ptr() as *mut i8, mask, chunk);
        // tokens.assume_init_ref();
        _mm512_mask_compressstoreu_epi8(indices.as_mut_ptr() as *mut i8, mask, a);
        // indices.assume_init_ref();
    }
}

// TODO
// #![feature(portable_simd)]

use std::arch::x86_64::{_mm512_loadu_epi8, _mm512_maskz_loadu_epi8};

#[target_feature(enable = "avx512f")] // For `_mm512_loadu_si512`.
#[target_feature(enable = "avx512bw")] // For `_mm512_maskz_loadu_epi8`.
fn test2(source: &[u8]) {
    source.chunks(64).for_each(|chunk| {
        let ptr = chunk.as_ptr();
        if chunk.len() == 64 {
            #[allow(unsafe_code)]
            unsafe {
                _mm512_loadu_epi8(ptr as *const i8); // TODO: `_mm512_loadu_si512` instead?
            }
        } else {
            #[allow(unsafe_code)]
            unsafe {
                _mm512_maskz_loadu_epi8(0, ptr as *const i8);
            }
        }
        println!("{:?}", chunk);
    });
}

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::{
    // intrinsics::simd,
    mem::MaybeUninit,
    // simd::{LaneCount, Mask, MaskElement, Simd, SupportedLaneCount, cmp::SimdPartialEq},
};

use seq_macro::seq;

macro_rules! args_count_down_to_zero_from {
    ($func:ident, $n:expr) => {
        seq!(N in 0..=$n { $func(#( $n - N, )*) })
    };
}

#[inline]
#[target_feature(enable = "sse2")] // For `_mm_set1_epi8` and `_mm_set_epi8`.
#[target_feature(enable = "avx512f")] // For `_kor_mask16`.
#[target_feature(enable = "avx512bw,avx512vl")] // For `_mm_cmpeq_epi8_mask`.
#[target_feature(enable = "avx512vbmi2,avx512vl")] // For `_mm_maskz_compress_epi8`.
fn chunk16(chunk: __m128i) -> (__m128i, __m128i) {
    let mask = _kor_mask16(
        _kor_mask16(
            _mm_cmpeq_epi8_mask(chunk, _mm_set1_epi8(b'[' as i8)),
            _mm_cmpeq_epi8_mask(chunk, _mm_set1_epi8(b']' as i8)),
        ),
        _kor_mask16(
            _mm_cmpeq_epi8_mask(chunk, _mm_set1_epi8(b'(' as i8)),
            _mm_cmpeq_epi8_mask(chunk, _mm_set1_epi8(b')' as i8)),
        ),
    );
    let tokens = _mm_maskz_compress_epi8(mask, chunk);
    let indices = _mm_maskz_compress_epi8(mask, args_count_down_to_zero_from!(_mm_set_epi8, 15));
    (tokens, indices)
}

#[inline]
#[target_feature(enable = "avx")] // For `_mm256_set1_epi8` and `_mm256_set_epi8`.
#[target_feature(enable = "avx512bw")] // For `_kor_mask32`.
#[target_feature(enable = "avx512bw,avx512vl")] // For `_mm256_cmpeq_epi8_mask`.
#[target_feature(enable = "avx512vbmi2,avx512vl")] // For `_mm256_maskz_compress_epi8`.
fn chunk32(chunk: __m256i) -> (__m256i, __m256i) {
    let mask = _kor_mask32(
        _kor_mask32(
            _mm256_cmpeq_epi8_mask(chunk, _mm256_set1_epi8(b'[' as i8)),
            _mm256_cmpeq_epi8_mask(chunk, _mm256_set1_epi8(b']' as i8)),
        ),
        _kor_mask32(
            _mm256_cmpeq_epi8_mask(chunk, _mm256_set1_epi8(b'(' as i8)),
            _mm256_cmpeq_epi8_mask(chunk, _mm256_set1_epi8(b')' as i8)),
        ),
    );
    let tokens = _mm256_maskz_compress_epi8(mask, chunk);
    let indices = _mm256_maskz_compress_epi8(mask, args_count_down_to_zero_from!(_mm256_set_epi8, 31));
    (tokens, indices)
}

#[inline]
#[target_feature(enable = "avx512f")] // For `_mm512_set1_epi8` and `_mm512_set_epi8`.
#[target_feature(enable = "avx512bw")] // For `_kor_mask64` and `_mm512_cmpeq_epi8_mask`.
#[target_feature(enable = "avx512vbmi2")] // For `_mm512_maskz_compress_epi8`.
fn chunk64(chunk: __m512i) -> (__m512i, __m512i) {
    let mask = _kor_mask64(
        _kor_mask64(
            _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'[' as i8)),
            _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b']' as i8)),
        ),
        _kor_mask64(
            _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b'(' as i8)),
            _mm512_cmpeq_epi8_mask(chunk, _mm512_set1_epi8(b')' as i8)),
        ),
    );
    let tokens = _mm512_maskz_compress_epi8(mask, chunk);
    let indices = _mm512_maskz_compress_epi8(mask, args_count_down_to_zero_from!(_mm512_set_epi8, 63));
    (tokens, indices)
}

// madvise(mapping, size, MADV_SEQUENTIAL);

// fn mask_true_indices<'a, T, const LANES: usize>(mask: Mask<T, LANES>) -> &'a [u8]
// where
//     T: MaskElement,
//     LaneCount<LANES>: SupportedLaneCount,
// {
//     let mut indices = MaybeUninit::<[u8; LANES]>::uninit();
//     let mask = mask.to_bitmask();
//     if is_x86_feature_detected!("avx512vbmi2") && is_x86_feature_detected!("popcnt") {
//         if is_x86_feature_detected!("avx512f") {
//             #[allow(unsafe_code)]
//             unsafe {
//                 mask_true_indices_avx512f(&mut indices, mask as __mmask64, a)
//             }
//         } else if is_x86_feature_detected!("avx512vl") {
//             if is_x86_feature_detected!("avx") {
//                 #[allow(unsafe_code)]
//                 unsafe {
//                     mask_true_indices_avx(&mut indices, mask as __mmask32, a)
//                 }
//             } else if is_x86_feature_detected!("sse2") {
//                 #[allow(unsafe_code)]
//                 unsafe {
//                     mask_true_indices_sse2(&mut indices, mask as __mmask16, a)
//                 }
//             } else {
//                 mask_true_indices_portable(mask)
//             }
//         } else {
//             mask_true_indices_portable(mask)
//         }
//     } else {
//         mask_true_indices_portable(mask)
//     }
// }

// TODO: Look at `mask.first_set()` and `mask.count_ones()`.
// TODO: `scatter_select()`.
// https://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
// fn mask_true_indices_portable<const LANES: usize>(indices: &mut MaybeUninit<[u8; LANES]>, mask: u64) -> &[u8]
// where
//     LaneCount<LANES>: SupportedLaneCount,
// {
//     let mut mask = mask;
//     let mut count = 0;
//     while mask != 0 {
//         indices[count] = mask.trailing_zeros() as u8;
//         mask &= mask - 1;
//     }
//     #[allow(unsafe_code)]
//     let indices = unsafe { indices.assume_init_ref() };
//     &indices[..count]
// }

// #[target_feature(enable = "sse2")] // For `_mm_set_epi8`.
// #[target_feature(enable = "avx512vbmi2,avx512vl")] // For `_mm_mask_compressstoreu_epi8`.
// #[target_feature(enable = "popcnt")] // For `_popcnt32`.
// fn mask_true_indices_sse2(indices: &mut MaybeUninit<[u8; 16]>, mask: __mmask16) -> &[u8] {
//     let a = args_count_down_to_zero_from!(_mm_set_epi8, 15);
//     #[allow(unsafe_code)]
//     let indices = unsafe {
//         _mm_mask_compressstoreu_epi8(indices.as_mut_ptr() as *mut i8, mask, a);
//         indices.assume_init_ref()
//     };
//     &indices[.._popcnt32(mask as i32) as usize]
// }

// #[target_feature(enable = "avx")] // For `_mm256_set_epi8`.
// #[target_feature(enable = "avx512vbmi2,avx512vl")] // For `_mm256_mask_compressstoreu_epi8`.
// #[target_feature(enable = "popcnt")] // For `_popcnt32`.
// fn mask_true_indices_avx(indices: &mut MaybeUninit<[u8; 32]>, mask: __mmask32) -> &[u8] {
//     let a = args_count_down_to_zero_from!(_mm256_set_epi8, 31);
//     #[allow(unsafe_code)]
//     let indices = unsafe {
//         _mm256_mask_compressstoreu_epi8(indices.as_mut_ptr() as *mut i8, mask, a);
//         indices.assume_init_ref()
//     };
//     &indices[.._popcnt32(mask as i32) as usize]
// }

// #[target_feature(enable = "avx512f")] // For `_mm512_set_epi8`.
// #[target_feature(enable = "avx512vbmi2")] // For `_mm512_mask_compressstoreu_epi8`.
// #[target_feature(enable = "popcnt")] // For `_popcnt64`.
// pub(crate) fn mask_true_indices_avx512f(indices: &mut MaybeUninit<[u8; 64]>, mask: __mmask64) -> &[u8] {
//     let a = args_count_down_to_zero_from!(_mm512_set_epi8, 63);
//     #[allow(unsafe_code)]
//     let indices = unsafe {
//         _mm512_mask_compressstoreu_epi8(indices.as_mut_ptr() as *mut i8, mask, a);
//         indices.assume_init_ref()
//     };
//     &indices[.._popcnt64(mask as i64) as usize]
// }

// https://stackoverflow.com/questions/74356480/efficiently-find-indices-of-1-bits-in-large-array-using-simd
// https://www.reddit.com/r/simd/comments/wqni47/deleted_by_user/
// https://github.com/syzygy1/Cfish/issues/204
// https://lemire.me/blog/2018/03/08/iterating-over-set-bits-quickly-simd-edition/

// https://github.com/Validark/Accelerated-Zig-Parser
// https://www.youtube.com/live/NM1FNB5nagk?si=02rNj5vppQdZ9A63

// const LANES: usize = 64;

// fn markdown_urls(source: &[u8]) -> Vec<&[u8]> {
//     source.chunks(LANES).for_each(|chunk| {
//         let chunk = Simd::<u8, LANES>::load_or_default(chunk);
//         mask_true_indices(chunk.simd_eq(Simd::splat(b'[')));
//         mask_true_indices(chunk.simd_eq(Simd::splat(b']')));
//         mask_true_indices(chunk.simd_eq(Simd::splat(b'(')));
//         mask_true_indices(chunk.simd_eq(Simd::splat(b')')));
//         mask_true_indices(chunk.simd_eq(Simd::splat(b'\n')));
//     });
//     vec![]
// }

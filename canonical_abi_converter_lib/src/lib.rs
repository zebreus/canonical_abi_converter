#![feature(adt_const_params)]
#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(str_from_utf16_endian)]
#![feature(macro_metavar_expr)]
#![feature(slice_as_array)]
#![feature(more_qualified_paths)]

pub use alignment::{alignment, alignment_flags, alignment_list, discriminant_type};
pub use element_size::{align_to, elem_size, elem_size_flags, elem_size_list};
pub use loading::{
    canonicalize_nan32, canonicalize_nan64, convert_i32_to_char, convert_int_to_bool,
    core_f32_reinterpret_i32, core_f64_reinterpret_i64, decode_i32_as_float, decode_i64_as_float,
    lift_async_value, lift_borrow, lift_error_context, lift_future, lift_own, lift_stream, load,
    load_list, load_list_from_range, load_list_from_valid_range, load_string,
    load_string_from_range,
};
pub use storing::{
    char_to_i32, core_i32_reinterpret_f32, core_i64_reinterpret_f64, encode_float_as_i32,
    encode_float_as_i64, lower_error_context, maybe_scramble_nan32, maybe_scramble_nan64,
    random_nan_bits, store, store_array, store_latin1_to_utf8, store_list,
    store_probably_utf16_to_latin1_or_utf16, store_string, store_string_to_latin1_or_utf16,
    store_utf8_to_utf16, store_utf16_to_utf8,
};
mod alignment;
mod element_size;
mod loading;
mod storing;

pub const fn max(current_max: u32, numbers: &[u32]) -> u32 {
    let Some((head, tail)) = numbers.split_first() else {
        return current_max;
    };
    if current_max > *head {
        return max(current_max, tail);
    } else {
        return max(*head, tail);
    }
}

#[derive(Debug, Clone)]
pub enum ConverterError {
    OutOfBoundsMemoryAccess {
        accessed: u32,
        size: u32,
    },
    SomethingWentWrong,
    CharOutOfRange {
        codepoint: i32,
    },
    CharIsSurrogate {
        codepoint: i32,
    },
    ValueForBoolOutOfRange,
    /// String pointer is not aligned
    StringAlignmentError,
    /// List pointer is not aligned
    ListAlignmentError,
    // TODO: Support for Latin1 encoded strings
    Latin1EncodedStringsNotSupported,
    // Invalid discriminant index
    InvalidDiscriminantIndex,
    /// String is too long to store
    StringTooLongToStore {
        length: u32,
    },
    /// List is too long too store
    ListTooLongToStore {
        /// Size in bytes
        size: u32,
    },
    /// Allocation return unaligned pointer
    AllocatedPointerNotAligned,
    /// Failed to allocate memory for a string
    AllocationFailedForAString,
    /// Failed to allocate memory for a list
    AllocationFailedForAList,
}

pub enum ContextStringEncoding {
    Utf8,
    Utf16,
    Latin1Utf16,
}
pub enum RealStringEncoding {
    Utf8,
    Utf16Le,
    Latin1,
}
impl RealStringEncoding {
    fn decode(&self, data: &[u8]) -> Result<String, ConverterError> {
        match self {
            RealStringEncoding::Utf8 => Ok(String::from_utf8_lossy(data).to_string()),
            RealStringEncoding::Utf16Le => Ok(String::from_utf16le_lossy(data)),
            RealStringEncoding::Latin1 => Err(ConverterError::Latin1EncodedStringsNotSupported),
        }
    }
}

pub trait Context {
    fn string_encoding(&self) -> ContextStringEncoding;
    fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError>;
    fn get_array_mut<const SIZE: usize>(
        &mut self,
        offset: u32,
    ) -> Result<&mut [u8; SIZE], ConverterError>;
    fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError>;
    fn get_slice_mut(&mut self, offset: u32, size: usize) -> Result<&mut [u8], ConverterError>;
    /// Reallocate the memory at the given offset
    /// Returns the new offset
    ///
    /// Can be used for freeing memory by passing 0 as the new size
    /// Can be used for allocating memory by passing 0 as the old offset
    fn realloc(
        &mut self,
        old_offset: u32,
        old_size: u32,
        align: u32,
        new_size: u32,
    ) -> Result<u32, ConverterError>;
}

pub trait CanonicalAbi
where
    Self: Sized,
{
    const ALIGNMENT: u32;
    const SIZE: u32;

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError>;
}

impl CanonicalAbi for () {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 0;
    fn load<'a, C: Context>(_context: &'a C, _offset: u32) -> Result<Self, ConverterError> {
        Ok(())
    }
}
impl CanonicalAbi for bool {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    // fn load(_data: &[u8; Self::SIZE]) -> Self {
    //     true
    // }
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        convert_int_to_bool(load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for u8 {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i8 {
    const ALIGNMENT: u32 = 1;
    const SIZE: u32 = 1;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u16 {
    const ALIGNMENT: u32 = 2;
    const SIZE: u32 = 2;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i16 {
    const ALIGNMENT: u32 = 2;
    const SIZE: u32 = 2;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for u64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for i64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_int!(context, offset, Self)
    }
}
impl CanonicalAbi for f32 {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        decode_i32_as_float(context, load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for f64 {
    const ALIGNMENT: u32 = 8;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        decode_i64_as_float(context, load_int!(context, offset, i64)?)
    }
}

impl CanonicalAbi for char {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 4;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        convert_i32_to_char(context, load_int!(context, offset, i32)?)
    }
}
impl CanonicalAbi for String {
    const ALIGNMENT: u32 = 4;
    const SIZE: u32 = 8;
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_string(context, offset)
    }
}

impl<T: CanonicalAbi> CanonicalAbi for Vec<T> {
    const ALIGNMENT: u32 = { alignment_list::<T, -1>() };
    const SIZE: u32 = { elem_size_list::<T, -1>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        load_list::<C, T, -1>(context, offset)
    }
}

pub struct WrappedBitflags<A, T: bitflags::Flags<Bits = A>> {
    pub inner: T,
}
impl<T: bitflags::Flags<Bits = u8>> CanonicalAbi for WrappedBitflags<u8, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}
impl<T: bitflags::Flags<Bits = u16>> CanonicalAbi for WrappedBitflags<u16, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}
impl<T: bitflags::Flags<Bits = u32>> CanonicalAbi for WrappedBitflags<u32, T> {
    const ALIGNMENT: u32 = { alignment_flags::<T::Bits>() };
    const SIZE: u32 = { elem_size_flags::<T::Bits>() };
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        Ok(WrappedBitflags {
            inner: T::from_bits_truncate(load_int!(context, offset, T::Bits)?),
        })
    }
}

impl<T: CanonicalAbi> CanonicalAbi for Option<T> {
    const ALIGNMENT: u32 = alignment_variant!(2, (), T);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);

    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let value = load_variant!(
            context,
            offset,
            enum Self {
                None,
                Some(T),
            }
        );
        Ok(value)
    }
}

impl<T: CanonicalAbi, E: CanonicalAbi> CanonicalAbi for Result<T, E> {
    const ALIGNMENT: u32 = alignment_variant!(2, T, E);
    const SIZE: u32 = elem_size_variant!(Self, 2, (), T);
    fn load<'a, C: Context>(context: &'a C, offset: u32) -> Result<Self, ConverterError> {
        let value = load_variant!(
            context,
            offset,
            enum Self {
                Ok(T),
                Err(E),
            }
        );
        Ok(value)
    }
}

macro_rules! impl_canonical_abi_for_tuple {
    ($($name:ident),*) => {
        impl<$($name: CanonicalAbi),*> CanonicalAbi for ($($name,)*) {
            const ALIGNMENT: u32 = alignment_record!($($name),*);
            const SIZE: u32 = elem_size_record!(Self, $($name),*);
            fn load<'a, CC: Context>(context: &'a CC, offset: u32) -> Result<Self, ConverterError> {
                Ok(load_tuple!(context, offset, ($($name,)*)))
            }
        }
    };
}

impl_canonical_abi_for_tuple!(A);
impl_canonical_abi_for_tuple!(A, B);
impl_canonical_abi_for_tuple!(A, B, C);
impl_canonical_abi_for_tuple!(A, B, C, D);
impl_canonical_abi_for_tuple!(A, B, C, D, E);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O);
impl_canonical_abi_for_tuple!(A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P);

#[cfg(test)]
mod tests {

    use super::*;
    use std::{
        alloc::{GlobalAlloc, Layout},
        convert::TryInto,
        ptr::null,
        vec,
    };

    const HEAP_SIZE: usize = 4096;

    #[derive()]
    pub struct SampleContext {
        memory: Vec<u8>,
        allocator: ::talc::Talck<::spin::Mutex<()>, ::talc::ClaimOnOom>,
    }

    impl std::fmt::Debug for SampleContext {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SampleContext")
                .field("memory", &self.memory)
                .finish()
        }
    }

    impl SampleContext {
        pub fn new(mut memory: Vec<u8>) -> Self {
            let original_size = memory.len();
            memory.extend_from_slice(&vec![0; HEAP_SIZE]);
            let heap_ptr = memory[original_size..].as_mut_ptr() as *mut [u8; HEAP_SIZE];
            let allocator = unsafe {
                ::talc::Talc::new(::talc::ClaimOnOom::new(::talc::Span::from_array(heap_ptr)))
                    .lock()
            };
            SampleContext { memory, allocator }
        }
        pub fn get_memory(&self) -> &[u8] {
            &self.memory
        }
    }

    impl Context for SampleContext {
        fn get_array<const SIZE: usize>(&self, offset: u32) -> Result<&[u8; SIZE], ConverterError> {
            if offset as usize + SIZE > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + SIZE) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + SIZE]
                .as_array::<SIZE>()
                .ok_or(ConverterError::SomethingWentWrong)
        }
        fn get_array_mut<const SIZE: usize>(
            &mut self,
            offset: u32,
        ) -> Result<&mut [u8; SIZE], ConverterError> {
            if offset as usize + SIZE > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + SIZE) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + SIZE]
                .as_mut_array::<SIZE>()
                .ok_or(ConverterError::SomethingWentWrong)
        }
        fn get_slice(&self, offset: u32, size: usize) -> Result<&[u8], ConverterError> {
            if offset as usize + size > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + size) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + size]
                .try_into()
                .map_err(|_| ConverterError::SomethingWentWrong)
        }
        fn get_slice_mut(&mut self, offset: u32, size: usize) -> Result<&mut [u8], ConverterError> {
            if offset as usize + size > self.memory.len() {
                return Err(ConverterError::OutOfBoundsMemoryAccess {
                    accessed: (offset as usize + size) as u32,
                    size: self.memory.len() as u32,
                });
            }
            self.memory[offset as usize..offset as usize + size]
                .as_mut()
                .try_into()
                .map_err(|_| ConverterError::SomethingWentWrong)
        }
        fn string_encoding(&self) -> ContextStringEncoding {
            ContextStringEncoding::Utf8
        }

        fn realloc(
            &mut self,
            old_offset: u32,
            old_size: u32,
            align: u32,
            new_size: u32,
        ) -> Result<u32, ConverterError> {
            let ptr = self.memory.as_mut_ptr();
            let result = match (old_offset, new_size) {
                (0, new_size) => unsafe {
                    let new_ptr = self.allocator.alloc(Layout::from_size_align_unchecked(
                        new_size as usize,
                        align as usize,
                    ));
                    new_ptr
                },
                (old_offset, 0) => unsafe {
                    let old_ptr = ptr.add(old_offset as usize);
                    self.allocator.dealloc(
                        old_ptr,
                        Layout::from_size_align_unchecked(old_size as usize, align as usize),
                    );
                    null()
                },
                (old_offset, new_size) => unsafe {
                    let old_ptr = ptr.add(old_offset as usize);
                    self.allocator.realloc(
                        old_ptr,
                        Layout::from_size_align_unchecked(old_size as usize, align as usize),
                        new_size as usize,
                    )
                },
            };
            if result.is_null() {
                return Ok(0);
            }
            return Ok(((result as usize) - (ptr as usize)) as u32);
        }
    }
}

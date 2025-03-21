//! This module contains the implementation of the loading section of the canonical ABI.
//!
//! The structure of the module is closely modeled after the reference python implementation from the [canonical ABI explainer](https://github.com/WebAssembly/component-model/blob/6e08e28368fe6301be956ece68d08077ebc09423/design/mvp/CanonicalABI.md#loading). We aim to have a equivalent implementation of most of the functions in the python implementation. Some will need to be implemented as macros, but the goal is to stay close to the python implementation. Every python function that does not have a direct equivalent, will be mentioned in a comment.

use crate::{
    CanonicalAbi, Context, ContextStringEncoding, ConverterError, RealStringEncoding, align_to,
};

/// The load function defines how to read a value of a given value type t out of linear memory starting at offset ptr, returning the value represented as a Python value. Presenting the definition of load piecewise, we start with the top-level case analysis:
///
/// ```python
/// def load(cx, ptr, t):
///   assert(ptr == align_to(ptr, alignment(t)))
///   assert(ptr + elem_size(t) <= len(cx.opts.memory))
///   match despecialize(t):
///     case BoolType()         : return convert_int_to_bool(load_int(cx, ptr, 1))
///     case U8Type()           : return load_int(cx, ptr, 1)
///     case U16Type()          : return load_int(cx, ptr, 2)
///     case U32Type()          : return load_int(cx, ptr, 4)
///     case U64Type()          : return load_int(cx, ptr, 8)
///     case S8Type()           : return load_int(cx, ptr, 1, signed = True)
///     case S16Type()          : return load_int(cx, ptr, 2, signed = True)
///     case S32Type()          : return load_int(cx, ptr, 4, signed = True)
///     case S64Type()          : return load_int(cx, ptr, 8, signed = True)
///     case F32Type()          : return decode_i32_as_float(load_int(cx, ptr, 4))
///     case F64Type()          : return decode_i64_as_float(load_int(cx, ptr, 8))
///     case CharType()         : return convert_i32_to_char(cx, load_int(cx, ptr, 4))
///     case StringType()       : return load_string(cx, ptr)
///     case ErrorContextType() : return lift_error_context(cx, load_int(cx, ptr, 4))
///     case ListType(t, l)     : return load_list(cx, ptr, t, l)
///     case RecordType(fields) : return load_record(cx, ptr, fields)
///     case VariantType(cases) : return load_variant(cx, ptr, cases)
///     case FlagsType(labels)  : return load_flags(cx, ptr, labels)
///     case OwnType()          : return lift_own(cx, load_int(cx, ptr, 4), t)
///     case BorrowType()       : return lift_borrow(cx, load_int(cx, ptr, 4), t)
///     case StreamType(t)      : return lift_stream(cx, load_int(cx, ptr, 4), t)
///     case FutureType(t)      : return lift_future(cx, load_int(cx, ptr, 4), t)
/// ```
///
/// The load functions is represented by the associated function `load` of `CanonicalAbi`. Because of that this function is basically a no-op.
pub fn load<'a, T: CanonicalAbi, C: Context>(
    context: &'a C,
    offset: u32,
) -> Result<T, ConverterError> {
    T::load(context, offset)
}

/// Integers are loaded directly from memory, with their high-order bit interpreted according to the signedness of the type.
///
/// ```python
/// def load_int(cx, ptr, nbytes, signed = False):
///   return int.from_bytes(cx.opts.memory[ptr : ptr+nbytes], 'little', signed = signed)
/// ```
///
/// This macro works a bit different than the python function. Instead of the size and signedness, it uses the type to determine the size and signedness of the integer.
#[macro_export]
macro_rules! load_int {
    ($context:expr, $offset:expr, $ty:ty) => {
        Ok(<$ty>::from_le_bytes(
            *($context.get_array::<{ <$ty>::SIZE as usize }>($offset)?),
        ))
    };
}

/// Integer-to-boolean conversions treats 0 as false and all other bit-patterns as true
///
/// ```python
/// def convert_int_to_bool(i):
///   assert(i >= 0)
///   return bool(i)
/// ```
pub fn convert_int_to_bool(i: i32) -> Result<bool, ConverterError> {
    if i < 0 {
        return Err(ConverterError::ValueForBoolOutOfRange);
    }
    return Ok(i != 0);
}

/// There is only one canonical NaN value for 32 bit floats
///
/// ```python
/// CANONICAL_FLOAT32_NAN = 0x7fc00000
/// ```
const CANONICAL_FLOAT32_NAN: u32 = 0x7fc00000;

/// There is only one canonical NaN value for 64 bit floats
///
/// ```python
/// CANONICAL_FLOAT64_NAN = 0x7ff8000000000000
/// ```
const CANONICAL_FLOAT64_NAN: u64 = 0x7ff8000000000000;

/// There is only one unique NaN value per floating-point type. This reflects the practical reality that some languages and protocols do not preserve these bits. In the Python code below, this is expressed as canonicalizing NaNs to a particular bit pattern.
///
/// ```python
/// def canonicalize_nan32(f):
///   if math.isnan(f):
///     f = core_f32_reinterpret_i32(CANONICAL_FLOAT32_NAN)
///     assert(math.isnan(f))
///   return f
/// ```
pub fn canonicalize_nan32(value: f32) -> f32 {
    if value.is_nan() {
        return f32::from_bits(CANONICAL_FLOAT32_NAN);
    }
    value
}

/// There is only one unique NaN value per floating-point type. This reflects the practical reality that some languages and protocols do not preserve these bits. In the Python code below, this is expressed as canonicalizing NaNs to a particular bit pattern.
///
/// ```python
/// def canonicalize_nan64(f):
///   if math.isnan(f):
///     f = core_f64_reinterpret_i64(CANONICAL_FLOAT64_NAN)
///     assert(math.isnan(f))
///   return f
/// ```
pub fn canonicalize_nan64(value: f64) -> f64 {
    if value.is_nan() {
        return f64::from_bits(CANONICAL_FLOAT64_NAN);
    }
    value
}

/// Floats are loaded directly from memory, with the sign and payload information of NaN values discarded. Consequently, there is only one unique NaN value per floating-point type. This reflects the practical reality that some languages and protocols do not preserve these bits. In the Python code below, this is expressed as canonicalizing NaNs to a particular bit pattern.
///
/// ```python
/// def decode_i32_as_float(i):
///   return canonicalize_nan32(core_f32_reinterpret_i32(i))
/// ```
pub fn decode_i64_as_float<C: Context>(
    _context: &C,
    codepoint: i64,
) -> Result<f64, ConverterError> {
    Ok(canonicalize_nan64(core_f64_reinterpret_i64(codepoint)))
}

/// Floats are loaded directly from memory, with the sign and payload information of NaN values discarded. Consequently, there is only one unique NaN value per floating-point type. This reflects the practical reality that some languages and protocols do not preserve these bits. In the Python code below, this is expressed as canonicalizing NaNs to a particular bit pattern.
///
/// ```python
/// def decode_i64_as_float(i):
///   return canonicalize_nan64(core_f64_reinterpret_i64(i))
/// ```
pub fn decode_i32_as_float<C: Context>(
    _context: &C,
    codepoint: i32,
) -> Result<f32, ConverterError> {
    Ok(canonicalize_nan32(core_f32_reinterpret_i32(codepoint)))
}

/// ```python
/// def core_f32_reinterpret_i32(i):
///   return struct.unpack('<f', struct.pack('<I', i))[0] # f32.reinterpret_i32
/// ```
pub fn core_f64_reinterpret_i64(value: i64) -> f64 {
    f64::from_bits(value as u64)
}
/// ```python
/// def core_f64_reinterpret_i64(i):
///   return struct.unpack('<d', struct.pack('<Q', i))[0] # f64.reinterpret_i64
/// ```
pub fn core_f32_reinterpret_i32(value: i32) -> f32 {
    f32::from_bits(value as u32)
}

/// An `i32` is converted to a `char` (a [Unicode Scalar Value]) by dynamically testing that its unsigned integral value is in the valid [Unicode Code Point] range and not a [Surrogate]:
///
/// ```python
/// def convert_i32_to_char(cx, i):
///   assert(i >= 0)
///   trap_if(i >= 0x110000)
///   trap_if(0xD800 <= i <= 0xDFFF)
///   return chr(i)
/// ```
pub fn convert_i32_to_char<C: Context>(
    _context: &C,
    codepoint: i32,
) -> Result<char, ConverterError> {
    if codepoint < 0 || codepoint > 0x10FFFF {
        return Err(ConverterError::CharOutOfRange);
    }
    if codepoint >= 0xD800 && codepoint <= 0xDFFF {
        return Err(ConverterError::CharIsSurrogate);
    }
    Ok(char::from_u32(codepoint as u32).ok_or(ConverterError::CharOutOfRange)?)
}

/// Strings are loaded from two `i32` values: a pointer (offset in linear memory) and a number of bytes. There are three supported string encodings in [`canonopt`]: [UTF-8], [UTF-16] and `latin1+utf16`. This last options allows a *dynamic* choice between [Latin-1] and UTF-16, indicated by the high bit of the second `i32`. String values include their original encoding and byte length as a "hint" that enables `store_string` (defined below) to make better up-front allocation size choices in many cases. Thus, the value produced by `load_string` isn't simply a Python `str`, but a *tuple* containing a `str`, the original encoding and the original byte length.
///
/// ```python
/// def load_string(cx, ptr) -> String:
///   begin = load_int(cx, ptr, 4)
///   tagged_code_units = load_int(cx, ptr + 4, 4)
///   return load_string_from_range(cx, begin, tagged_code_units)
/// ```
pub fn load_string<C: Context>(_context: &C, offset: u32) -> Result<String, ConverterError> {
    let begin = u32::from_le_bytes(*_context.get_array::<4>(offset as u32)?);
    let tagged_code_units = u32::from_le_bytes(*_context.get_array::<4>(offset + 4)?);
    load_string_from_range(_context, begin, tagged_code_units)
}

/// ```python
/// UTF16_TAG = 1 << 31
/// ```
const UTF16_TAG: u32 = 1 << 31;

/// ```python
/// def load_string_from_range(cx, ptr, tagged_code_units) -> String:
///   match cx.opts.string_encoding:
///     case 'utf8':
///       alignment = 1
///       byte_length = tagged_code_units
///       encoding = 'utf-8'
///     case 'utf16':
///       alignment = 2
///       byte_length = 2 * tagged_code_units
///       encoding = 'utf-16-le'
///     case 'latin1+utf16':
///       alignment = 2
///       if bool(tagged_code_units & UTF16_TAG):
///         byte_length = 2 * (tagged_code_units ^ UTF16_TAG)
///         encoding = 'utf-16-le'
///       else:
///         byte_length = tagged_code_units
///         encoding = 'latin-1'
///
///   trap_if(ptr != align_to(ptr, alignment))
///   trap_if(ptr + byte_length > len(cx.opts.memory))
///   try:
///     s = cx.opts.memory[ptr : ptr+byte_length].decode(encoding)
///   except UnicodeError:
///     trap()
///
///   return (s, cx.opts.string_encoding, tagged_code_units)
/// ```
pub fn load_string_from_range<C: Context>(
    _context: &C,
    offset: u32,
    tagged_code_units: u32,
) -> Result<String, ConverterError> {
    let (alignment, byte_length, encoding) = match _context.string_encoding() {
        ContextStringEncoding::Utf8 => (1, tagged_code_units, RealStringEncoding::Utf8),

        ContextStringEncoding::Utf16 => (2, 2 * tagged_code_units, RealStringEncoding::Utf16Le),
        ContextStringEncoding::Latin1Utf16 => (
            2,
            if tagged_code_units & UTF16_TAG != 0 {
                2 * (tagged_code_units ^ UTF16_TAG)
            } else {
                tagged_code_units
            },
            if tagged_code_units & UTF16_TAG != 0 {
                RealStringEncoding::Utf16Le
            } else {
                RealStringEncoding::Latin1
            },
        ),
    };
    if offset != align_to(offset, alignment) {
        return Err(ConverterError::StringAlignmentError);
    }
    let data = _context.get_slice(offset, byte_length as usize)?;

    encoding.decode(data)
}

/// Error context values are lifted directly from the per-component-instance error_contexts table:
///
/// ```python
/// def lift_error_context(cx, i):
///   return cx.inst.error_contexts.get(i)
/// ```
pub fn lift_error_context<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Error contexts are not in scope for now");
}

/// Lists and records are loaded by recursively loading their elements/fields
///
/// ```python
/// def load_list(cx, ptr, elem_type, maybe_length):
///   if maybe_length is not None:
///     return load_list_from_valid_range(cx, ptr, maybe_length, elem_type)
///   begin = load_int(cx, ptr, 4)
///   length = load_int(cx, ptr + 4, 4)
///   return load_list_from_range(cx, begin, length, elem_type)
/// ```
pub fn load_list<'a, C: Context, T: CanonicalAbi, const LENGTH: i32>(
    context: &'a C,
    offset: u32,
) -> Result<Vec<T>, ConverterError> {
    if LENGTH > 0 {
        return load_list_from_valid_range(context, offset, LENGTH as u32);
    }
    let begin = load_int!(context, offset, u32)?;
    let length = load_int!(context, offset + 4, u32)?;
    return load_list_from_range(context, begin, length);
}

/// ```python
/// def load_list_from_range(cx, ptr, length, elem_type):
///   trap_if(ptr != align_to(ptr, alignment(elem_type)))
///   trap_if(ptr + length * elem_size(elem_type) > len(cx.opts.memory))
///   return load_list_from_valid_range(cx, ptr, length, elem_type)
/// ```
///
/// The check if the pointer is in bounds is always done, when loading data from memory, so we can skip it here.
pub fn load_list_from_range<'a, C: Context, T: CanonicalAbi>(
    context: &'a C,
    offset: u32,
    length: u32,
) -> Result<Vec<T>, ConverterError> {
    if offset != align_to(offset, T::ALIGNMENT) {
        return Err(ConverterError::ListAlignmentError);
    }
    load_list_from_valid_range(context, offset, length)
}

/// ```python
/// def load_list_from_valid_range(cx, ptr, length, elem_type):
///   a = []
///   for i in range(length):
///     a.append(load(cx, ptr + i * elem_size(elem_type), elem_type))
///   return a
/// ```
pub fn load_list_from_valid_range<'a, C: Context, T: CanonicalAbi>(
    context: &'a C,
    offset: u32,
    length: u32,
) -> Result<Vec<T>, ConverterError> {
    let mut result = Vec::with_capacity(length as usize);
    let mut current_offset = offset;
    for _ in 0..length {
        result.push(T::load(context, current_offset)?);
        current_offset += T::SIZE;
    }
    Ok(result)
}

/// ```python
/// def load_record(cx, ptr, fields):
///   record = {}
///   for field in fields:
///     ptr = align_to(ptr, alignment(field.t))
///     record[field.label] = load(cx, ptr, field.t)
///     ptr += elem_size(field.t)
///   return record
/// ```
///
/// There also is `load_tuple` for tuples, because we cant treat them like structs in rust.
///
/// Needs to be used in a context where a `ConverterError` can be returned.
#[macro_export]
macro_rules! load_record {
    ($context:expr, $offset:expr, struct $self:path { $($name:ident : $typ:ty,)* }) => {
        {
            let offset = $offset;
             $self { $($name: {
                let offset = align_to(offset, <$typ>::ALIGNMENT);
                let v = <$typ>::load($context, offset)?;
                #[allow(unused)]
                let offset = offset + <$typ>::SIZE;
                v
            }),* }
        }
    };
}

/// This functions does not have a direct python equivalent, but it does the same as `load_record` but for tuples.
///
/// Needs to be used in a context where a `ConverterError` can be returned.
#[macro_export]
macro_rules! load_tuple {
    ($context:expr, $offset:expr, ( $($typ:ty,)* )) => {
        {
            let offset = $offset;
            ( $({
                let offset = align_to(offset, <$typ>::ALIGNMENT);
                let v = <$typ>::load($context, offset)?;
                #[allow(unused)]
                let offset = offset + <$typ>::SIZE;
                v
            },)* )
        }
    };
}

/// Variants are loaded using the order of the cases in the type to determine the case index, assigning 0 to the first case, 1 to the next case, etc. While the code below appears to perform case-label lookup at runtime, a normal implementation can build the appropriate index tables at compile-time so that variant-passing is always O(1) and not involving string operations.
///
/// ```python
/// def load_variant(cx, ptr, cases):
///   disc_size = elem_size(discriminant_type(cases))
///   case_index = load_int(cx, ptr, disc_size)
///   ptr += disc_size
///   trap_if(case_index >= len(cases))
///   c = cases[case_index]
///   ptr = align_to(ptr, max_case_alignment(cases))
///   if c.t is None:
///     return { c.label: None }
///   return { c.label: load(cx, ptr, c.t) }
/// ```
///
/// Needs to be used in a context where a `ConverterError` can be returned.
#[macro_export]
macro_rules! load_variant {
    ($context:expr, $offset:expr, enum $self:path { $($name:ident $(( $typ:ty ))?,)* }) => {
        {
            let offset = $offset;
            const LENGTH: usize = ${count($name)};
            let discriminant_size = discriminant_type::<LENGTH>();
            let case_index = match discriminant_size {
                1 => load_int!($context, offset, u8)? as u32,
                2 => load_int!($context, offset, u16)? as u32,
                4 => load_int!($context, offset, u32)?,
                _ => panic!(),
            };
            let offset = offset + discriminant_size;

            match case_index {
                $(${index()} => Ok(<$self>::$name$((<$typ>::load($context, offset)?))?),)*
                _ => Err(ConverterError::InvalidDiscriminantIndex),
            }?
        }
    };
}

/// Flags are converted from a bit-vector to a dictionary whose keys are derived from the ordered labels of the flags type.
///
/// ```python
/// def load_flags(cx, ptr, labels):
///   i = load_int(cx, ptr, elem_size_flags(labels))
///   return unpack_flags_from_int(i, labels)
/// ```
#[macro_export]
macro_rules! load_flags {
    ($context:expr, $offset:expr, bitflags::Flags<Bits = $ty:ty>) => {{
        let value = load_int!($context, $offset, $ty)?;
        Ok(unpack_flags_from_int!(value, bitflags::Flags<Bits = $ty>))
    }};
}

/// ```python
/// def unpack_flags_from_int(i, labels):
///   record = {}
///   for l in labels:
///     record[l] = bool(i & 1)
///     i >>= 1
///   return record
/// ```
#[macro_export]
macro_rules! unpack_flags_from_int {
    ($value:expr, bitflags::Flags<Bits = $ty:ty>) => {{
        WrappedBitflags {
            inner: $ty::from_bits_truncate($value),
        }
    }};
}

/// own handles are lifted by removing the handle from the current component instance's handle table, so that ownership is transferred to the lowering component. The lifting operation fails if unique ownership of the handle isn't possible, for example if the index was actually a borrow or if the own handle is currently being lent out as borrows.
///
/// ```python
/// def lift_own(cx, i, t):
///   h = cx.inst.resources.remove(i)
///   trap_if(h.rt is not t.rt)
///   trap_if(h.num_lends != 0)
///   trap_if(not h.own)
///   return h.rep
/// ```
pub fn lift_own<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Own types are not in scope for now");
}

/// In contrast to own, borrow handles are lifted by reading the representation from the source handle, leaving the source handle intact in the current component instance's handle table:
///
/// ```python
/// def lift_borrow(cx, i, t):
///   assert(isinstance(cx.borrow_scope, Subtask))
///   h = cx.inst.resources.get(i)
///   trap_if(h.rt is not t.rt)
///   cx.borrow_scope.add_lender(h)
///   return h.rep
/// ```
pub fn lift_borrow<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Borrow types are not in scope for now");
}

/// Streams and futures are lifted in almost the same way, with the only difference being that it is a dynamic error to attempt to lift a future that has already been successfully read (which will leave it closed()):
///
/// ```python
/// def lift_stream(cx, i, t):
///   return lift_async_value(ReadableStreamEnd, WritableStreamEnd, cx, i, t)
/// ```
pub fn lift_stream<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Streams are not in scope for now");
}

/// ```python
/// def lift_future(cx, i, t):
///   v = lift_async_value(ReadableFutureEnd, WritableFutureEnd, cx, i, t)
///   trap_if(v.closed())
///   return v
/// ```
pub fn lift_future<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Futures are not in scope for now");
}

/// ```python
/// def lift_async_value(ReadableEndT, WritableEndT, cx, i, t):
///   assert(not contains_borrow(t))
///   e = cx.inst.waitables.get(i)
///   match e:
///     case ReadableEndT():
///       trap_if(e.copying)
///       cx.inst.waitables.remove(i)
///     case WritableEndT():
///       trap_if(e.paired)
///       e.paired = True
///     case _:
///       trap()
///   trap_if(e.stream.t != t)
///   return e.stream
/// ```
pub fn lift_async_value<C: Context>(_context: &C, _index: i32) -> Result<(), ConverterError> {
    todo!("Async values are not in scope for now");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::SampleContext;

    #[test]
    fn test_load_nothing() {
        let context = SampleContext::new(vec![]);
        let value: () = <()>::load(&context, 0).unwrap();
        assert_eq!(value, ());
    }

    #[test]
    fn test_load_bool() {
        let context = SampleContext::new(vec![0x01, 0x00, 0x00, 0x00]);
        let value: bool = bool::load(&context, 0).unwrap();
        assert_eq!(value, true);
    }

    #[test]
    fn test_load_u8() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: u8 = u8::load(&context, 0).unwrap();
        assert_eq!(value, 0x01);
    }

    #[test]
    fn test_load_i8() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: i8 = i8::load(&context, 0).unwrap();
        assert_eq!(value, 0x01);
    }

    #[test]
    fn test_load_u16() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: u16 = u16::load(&context, 0).unwrap();
        assert_eq!(value, 0x0201);
    }

    #[test]
    fn test_load_i16() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: i16 = i16::load(&context, 0).unwrap();
        assert_eq!(value, 0x0201);
    }

    #[test]
    fn test_load_u32() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: u32 = u32::load(&context, 0).unwrap();
        assert_eq!(value, 0x04030201);
    }

    #[test]
    fn test_load_i32() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04]);
        let value: i32 = i32::load(&context, 0).unwrap();
        assert_eq!(value, 0x04030201);
    }

    #[test]
    fn test_load_u64() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        let value: u64 = u64::load(&context, 0).unwrap();
        assert_eq!(value, 0x0807060504030201);
    }

    #[test]
    fn test_load_i64() {
        let context = SampleContext::new(vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
        let value: i64 = i64::load(&context, 0).unwrap();
        assert_eq!(value, 0x0807060504030201);
    }

    #[test]
    fn test_load_f32() {
        let context = SampleContext::new(vec![0x00, 0x00, 0x00, 0x00]);
        let value: f32 = f32::load(&context, 0).unwrap();
        assert_eq!(value, 0.0);
    }

    #[test]
    fn test_load_f64() {
        let context = SampleContext::new(123.45678f64.to_le_bytes().to_vec());
        let value: f64 = f64::load(&context, 0).unwrap();
        assert_eq!(value, 123.45678);
    }

    #[test]
    fn test_load_char() {
        let context = SampleContext::new(vec![0x41, 0x00, 0x00, 0x00]);
        let value: char = char::load(&context, 0).unwrap();
        assert_eq!(value, 'A');
    }

    #[test]
    fn test_load_string() {
        let context = SampleContext::new(vec![
            0x08, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x41, 0x42, 0x43, 0x44,
        ]);
        let value: String = String::load(&context, 0).unwrap();
        assert_eq!(value, "ABCD");
    }
}

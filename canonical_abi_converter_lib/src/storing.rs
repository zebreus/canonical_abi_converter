//! This module contains the implementation of the storing section of the canonical ABI.
//!
//! The structure of the module is closely modeled after the reference python implementation from the [canonical ABI explainer](https://github.com/WebAssembly/component-model/blob/6e08e28368fe6301be956ece68d08077ebc09423/design/mvp/CanonicalABI.md#storing). We aim to have a equivalent implementation of most of the functions in the python implementation. Some will need to be implemented as macros, but the goal is to stay close to the python implementation. Every python function that does not have a direct equivalent, will be mentioned in a comment.

use crate::{
    CanonicalAbi, Context, ContextStringEncoding, ConverterError, RealStringEncoding, align_to,
};

/// The `store` function defines how to write a value `v` of a given value type `t` into linear memory starting at offset `ptr`. Presenting the definition of `store` piecewise, we start with the top-level case analysis:
///
/// ```python
/// def store(cx, v, t, ptr):
///   assert(ptr == align_to(ptr, alignment(t)))
///   assert(ptr + elem_size(t) <= len(cx.opts.memory))
///   match despecialize(t):
///     case BoolType()         : store_int(cx, int(bool(v)), ptr, 1)
///     case U8Type()           : store_int(cx, v, ptr, 1)
///     case U16Type()          : store_int(cx, v, ptr, 2)
///     case U32Type()          : store_int(cx, v, ptr, 4)
///     case U64Type()          : store_int(cx, v, ptr, 8)
///     case S8Type()           : store_int(cx, v, ptr, 1, signed = True)
///     case S16Type()          : store_int(cx, v, ptr, 2, signed = True)
///     case S32Type()          : store_int(cx, v, ptr, 4, signed = True)
///     case S64Type()          : store_int(cx, v, ptr, 8, signed = True)
///     case F32Type()          : store_int(cx, encode_float_as_i32(v), ptr, 4)
///     case F64Type()          : store_int(cx, encode_float_as_i64(v), ptr, 8)
///     case CharType()         : store_int(cx, char_to_i32(v), ptr, 4)
///     case StringType()       : store_string(cx, v, ptr)
///     case ErrorContextType() : store_int(cx, lower_error_context(cx, v), ptr, 4)
///     case ListType(t, l)     : store_list(cx, v, ptr, t, l)
///     case RecordType(fields) : store_record(cx, v, ptr, fields)
///     case VariantType(cases) : store_variant(cx, v, ptr, cases)
///     case FlagsType(labels)  : store_flags(cx, v, ptr, labels)
///     case OwnType()          : store_int(cx, lower_own(cx, v, t), ptr, 4)
///     case BorrowType()       : store_int(cx, lower_borrow(cx, v, t), ptr, 4)
///     case StreamType(t)      : store_int(cx, lower_stream(cx, v, t), ptr, 4)
///     case FutureType(t)      : store_int(cx, lower_future(cx, v, t), ptr, 4)
/// ```
///
/// The store function is represented by the associated function `store` of `CanonicalAbi`. Because of that this function is basically a no-op.
pub fn store<'a, T: CanonicalAbi, C: Context>(
    _context: &mut C,
    _offset: u32,
    _value: &T,
) -> Result<(), ConverterError> {
    todo!("Implement store function on CanonicalAbi")
}

/// Integers are stored directly into memory. Because the input domain is exactly the integers in range for the given type, no extra range checks are necessary; the `signed` parameter is only present to ensure that the internal range checks of `int.to_bytes` are satisfied.
///
/// ```python
/// def store_int(cx, v, ptr, nbytes, signed = False):
///   cx.opts.memory[ptr : ptr+nbytes] = int.to_bytes(v, nbytes, 'little', signed = signed)
/// ```
#[macro_export]
macro_rules! store_int {
    ($context:expr, $offset:expr, $ty:ty, $value:expr) => {{
        let array = $context.get_array_mut::<{ <$ty>::SIZE as usize }>($offset)?;
        let value: $ty = $value;
        *array = value.to_le_bytes();
    }};
}

/// Floats are stored directly into memory, with the sign and payload bits of NaN values modified non-deterministically. This reflects the practical reality that different languages, protocols and CPUs have different effects on NaNs.
///
/// Although this non-determinism is expressed in the Python code below as generating a "random" NaN bit-pattern, native implementations do not need to use the same "random" algorithm, or even any random algorithm at all. Hosts may instead chose to canonicalize to an arbitrary fixed NaN value, or even to the original value of the NaN before lifting, allowing them to optimize away both the canonicalization of lifting and the randomization of lowering.
///
/// When a host implements the [deterministic profile], NaNs are canonicalized to a particular NaN bit-pattern.
///
/// ```python
/// def maybe_scramble_nan32(f):
///   if math.isnan(f):
///     if DETERMINISTIC_PROFILE:
///       f = core_f32_reinterpret_i32(CANONICAL_FLOAT32_NAN)
///     else:
///       f = core_f32_reinterpret_i32(random_nan_bits(32, 8))
///     assert(math.isnan(f))
///   return f
/// ```
///
/// NaN scrambling is not implemented in this library, because it is only relevant for the reference implementation of the canonical ABI.
pub fn maybe_scramble_nan32(f: f32) -> f32 {
    f
}

/// ```python
/// def maybe_scramble_nan64(f):
///   if math.isnan(f):
///     if DETERMINISTIC_PROFILE:
///       f = core_f64_reinterpret_i64(CANONICAL_FLOAT64_NAN)
///     else:
///       f = core_f64_reinterpret_i64(random_nan_bits(64, 11))
///     assert(math.isnan(f))
///   return f
/// ```
///
/// NaN scrambling is not implemented in this library, because it is only relevant for the reference implementation of the canonical ABI.
pub fn maybe_scramble_nan64(f: f64) -> f64 {
    f
}

/// ```python
/// def random_nan_bits(total_bits, exponent_bits):
///   fraction_bits = total_bits - exponent_bits - 1
///   bits = random.getrandbits(total_bits)
///   bits |= ((1 << exponent_bits) - 1) << fraction_bits
///   bits |= 1 << random.randrange(fraction_bits - 1)
///   return bits
/// ```
///
/// NaN scrambling is not implemented in this library, because it is only relevant for the reference implementation of the canonical ABI.
pub fn random_nan_bits(_total_bits: u32, _exponent_bits: u32) -> u64 {
    return 0u64;
}

/// ```python
/// def encode_float_as_i32(f):
///   return core_i32_reinterpret_f32(maybe_scramble_nan32(f))
/// ```
pub fn encode_float_as_i32(f: f32) -> i32 {
    return core_i32_reinterpret_f32(maybe_scramble_nan32(f));
}

/// ```python
/// def encode_float_as_i64(f):
///   return core_i64_reinterpret_f64(maybe_scramble_nan64(f))
/// ```
pub fn encode_float_as_i64(f: f64) -> i64 {
    return core_i64_reinterpret_f64(maybe_scramble_nan64(f));
}

/// ```python
/// def core_i32_reinterpret_f32(f):
///   return struct.unpack('<I', struct.pack('<f', f))[0] # i32.reinterpret_f32
/// ```
pub fn core_i32_reinterpret_f32(f: f32) -> i32 {
    return f.to_bits() as i32;
}

/// ```python
/// def core_i64_reinterpret_f64(f):
///   return struct.unpack('<Q', struct.pack('<d', f))[0] # i64.reinterpret_f64
/// ```
pub fn core_i64_reinterpret_f64(f: f64) -> i64 {
    return f.to_bits() as i64;
}

/// The integral value of a `char` (a [Unicode Scalar Value]) is a valid unsigned `i32` and thus no runtime conversion or checking is necessary:
///
/// ```python
/// def char_to_i32(c):
///   i = ord(c)
///   assert(0 <= i <= 0xD7FF or 0xD800 <= i <= 0x10FFFF)
///   return i
/// ```
pub fn char_to_i32(c: char) -> Result<i32, ConverterError> {
    let codepoint = c as i32;
    if codepoint < 0 || codepoint > 0x10FFFF {
        // Should never happen, because char is always a valid unicode scalar value
        return Err(ConverterError::CharOutOfRange { codepoint });
    }
    if codepoint >= 0xD800 && codepoint <= 0xDFFF {
        return Err(ConverterError::CharIsSurrogate { codepoint });
    }
    return Ok(codepoint);
}

/// Storing strings is complicated by the goal of attempting to optimize the different transcoding cases. In particular, one challenge is choosing the linear memory allocation size *before* examining the contents of the string. The reason for this constraint is that, in some settings where single-pass iterators are involved (host calls and post-MVP [adapter functions]), examining the contents of a string more than once would require making an engine-internal temporary copy of the whole string, which the component model specifically aims not to do. To avoid multiple passes, the canonical ABI instead uses a `realloc` approach to update the allocation size during the single copy. A blind `realloc` approach would normally suffer from multiple reallocations per string (e.g., using the standard doubling-growth strategy). However, as already shown in `load_string` above, string values come with two useful hints: their original encoding and byte length. From this hint data, `store_string` can do a much better job minimizing the number of reallocations.
///
/// We start with a case analysis to enumerate all the meaningful encoding combinations, subdividing the `latin1+utf16` encoding into either `latin1` or `utf16` based on the `UTF16_BIT` flag set by `load_string`:
///
/// ```python
/// def store_string(cx, v: String, ptr):
///   begin, tagged_code_units = store_string_into_range(cx, v)
///   store_int(cx, begin, ptr, 4)
///   store_int(cx, tagged_code_units, ptr + 4, 4)
/// ```
pub fn store_string<'a, C: Context>(
    context: &mut C,
    offset: u32,
    value: &str,
) -> Result<(), ConverterError> {
    let (begin, tagged_code_units) = store_string_into_range(context, value)?;
    store_int!(context, offset, u32, begin);
    store_int!(context, offset + 4, u32, tagged_code_units);
    Ok(())
}

/// ```python
/// def store_string_into_range(cx, v: String):
///   src, src_encoding, src_tagged_code_units = v
///
///   if src_encoding == 'latin1+utf16':
///     if bool(src_tagged_code_units & UTF16_TAG):
///       src_simple_encoding = 'utf16'
///       src_code_units = src_tagged_code_units ^ UTF16_TAG
///     else:
///       src_simple_encoding = 'latin1'
///       src_code_units = src_tagged_code_units
///   else:
///     src_simple_encoding = src_encoding
///     src_code_units = src_tagged_code_units
///   match cx.opts.string_encoding:
///     case 'utf8':
///       match src_simple_encoding:
///         case 'utf8'         : return store_string_copy(cx, src, src_code_units, 1, 1, 'utf-8')
///         case 'utf16'        : return store_utf16_to_utf8(cx, src, src_code_units)
///         case 'latin1'       : return store_latin1_to_utf8(cx, src, src_code_units)
///     case 'utf16':
///       match src_simple_encoding:
///         case 'utf8'         : return store_utf8_to_utf16(cx, src, src_code_units)
///         case 'utf16'        : return store_string_copy(cx, src, src_code_units, 2, 2, 'utf-16-le')
///         case 'latin1'       : return store_string_copy(cx, src, src_code_units, 2, 2, 'utf-16-le')
///     case 'latin1+utf16':
///       match src_encoding:
///         case 'utf8'         : return store_string_to_latin1_or_utf16(cx, src, src_code_units)
///         case 'utf16'        : return store_string_to_latin1_or_utf16(cx, src, src_code_units)
///         case 'latin1+utf16' :
///           match src_simple_encoding:
///             case 'latin1'   : return store_string_copy(cx, src, src_code_units, 1, 2, 'latin-1')
///             case 'utf16'    : return store_probably_utf16_to_latin1_or_utf16(cx, src, src_code_units)
/// ```
///
/// Ok, so for now we only support UTF-8 encoding, so we can simplify the function to only support that encoding. Our strings also do not contain any of the additional information that the reference implementation uses, so our function is really simple
pub fn store_string_into_range<'a, C: Context>(
    context: &mut C,
    value: &str,
) -> Result<(u32, u32), ConverterError> {
    let ContextStringEncoding::Utf8 = context.string_encoding() else {
        panic!("Storing strings is only supported with UTF-8 encoding for now");
    };
    // let src_simple_encoding = RealStringEncoding::Utf8;

    return store_string_copy(
        context,
        value,
        value.len() as u32,
        1,
        1,
        RealStringEncoding::Utf8,
    );
}

/// The choice of `MAX_STRING_BYTE_LENGTH` constant ensures that the high bit of a string's byte length is never set, keeping it clear for `UTF16_BIT`.
///
/// ```python
/// MAX_STRING_BYTE_LENGTH = (1 << 31) - 1
/// ```
const MAX_STRING_BYTE_LENGTH: i32 = i32::MAX;

/// The simplest 4 cases above can compute the exact destination size and then copy with a simply loop (that possibly inflates Latin-1 to UTF-16 by injecting a 0 byte after every Latin-1 byte).
///
/// ```python
/// def store_string_copy(cx, src, src_code_units, dst_code_unit_size, dst_alignment, dst_encoding):
///   dst_byte_length = dst_code_unit_size * src_code_units
///   trap_if(dst_byte_length > MAX_STRING_BYTE_LENGTH)
///   ptr = cx.opts.realloc(0, 0, dst_alignment, dst_byte_length)
///   trap_if(ptr != align_to(ptr, dst_alignment))
///   trap_if(ptr + dst_byte_length > len(cx.opts.memory))
///   encoded = src.encode(dst_encoding)
///   assert(dst_byte_length == len(encoded))
///   cx.opts.memory[ptr : ptr+len(encoded)] = encoded
///   return (ptr, src_code_units)
/// ```
pub fn store_string_copy<'a, C: Context>(
    context: &mut C,
    src: &str,
    src_code_units: u32,
    dst_code_unit_size: u32,
    dst_alignment: u32,
    _dst_encoding: RealStringEncoding,
) -> Result<(u32, u32), ConverterError> {
    let dst_byte_length = dst_code_unit_size * src_code_units;
    if dst_byte_length > MAX_STRING_BYTE_LENGTH as u32 {
        return Err(ConverterError::StringTooLongToStore {
            length: dst_byte_length,
        });
    }
    let ptr = context.realloc(0, 0, 1, dst_byte_length as u32)?;
    if ptr == 0 {
        return Err(ConverterError::AllocationFailedForAString);
    }
    if ptr != align_to(ptr, dst_alignment) {
        return Err(ConverterError::AllocatedPointerNotAligned);
    }
    let slice = context.get_slice_mut(ptr, dst_byte_length as usize)?;
    slice.copy_from_slice(src.as_bytes());
    return Ok((ptr, src.len() as u32));
}

/// The 2 cases of transcoding into UTF-8 share an algorithm that starts by optimistically assuming that each code unit of the source string fits in a single UTF-8 byte and then, failing that, reallocates to a worst-case size, finishes the copy, and then finishes with a shrinking reallocation.
///
/// ```python
/// def store_utf16_to_utf8(cx, src, src_code_units):
///   worst_case_size = src_code_units * 3
///   return store_string_to_utf8(cx, src, src_code_units, worst_case_size)
/// ```
pub fn store_utf16_to_utf8<'a, C: Context>(
    context: &mut C,
    src: &str,
    src_code_units: u32,
) -> Result<(u32, u32), ConverterError> {
    let worst_case_size = src_code_units * 3;
    return store_string_to_utf8(context, src, src_code_units, worst_case_size);
}

/// ```python
/// def store_latin1_to_utf8(cx, src, src_code_units):
///   worst_case_size = src_code_units * 2
///   return store_string_to_utf8(cx, src, src_code_units, worst_case_size)
/// ```
pub fn store_latin1_to_utf8<'a, C: Context>(
    context: &mut C,
    src: &str,
    src_code_units: u32,
) -> Result<(u32, u32), ConverterError> {
    let worst_case_size = src_code_units * 2;
    return store_string_to_utf8(context, src, src_code_units, worst_case_size);
}

/// ```python
/// def store_string_to_utf8(cx, src, src_code_units, worst_case_size):
///   assert(src_code_units <= MAX_STRING_BYTE_LENGTH)
///   ptr = cx.opts.realloc(0, 0, 1, src_code_units)
///   trap_if(ptr + src_code_units > len(cx.opts.memory))
///   for i,code_point in enumerate(src):
///     if ord(code_point) < 2**7:
///       cx.opts.memory[ptr + i] = ord(code_point)
///     else:
///       trap_if(worst_case_size > MAX_STRING_BYTE_LENGTH)
///       ptr = cx.opts.realloc(ptr, src_code_units, 1, worst_case_size)
///       trap_if(ptr + worst_case_size > len(cx.opts.memory))
///       encoded = src.encode('utf-8')
///       cx.opts.memory[ptr+i : ptr+len(encoded)] = encoded[i : ]
///       if worst_case_size > len(encoded):
///         ptr = cx.opts.realloc(ptr, worst_case_size, 1, len(encoded))
///         trap_if(ptr + len(encoded) > len(cx.opts.memory))
///       return (ptr, len(encoded))
///   return (ptr, src_code_units)
/// ```
pub fn store_string_to_utf8<'a, C: Context>(
    _context: &mut C,
    _src: &str,
    _src_code_units: u32,
    _worst_case_size: u32,
) -> Result<(u32, u32), ConverterError> {
    todo!(
        "`store_string_to_utf8` is not implemented yet because encodings other than UTF-8 are not in scope"
    );
}

/// Converting from UTF-8 to UTF-16 performs an initial worst-case size allocation (assuming each UTF-8 byte encodes a whole code point that inflates into a two-byte UTF-16 code unit) and then does a shrinking reallocation at the end if multiple UTF-8 bytes were collapsed into a single 2-byte UTF-16 code unit:
///
/// ```python
/// def store_utf8_to_utf16(cx, src, src_code_units):
///   worst_case_size = 2 * src_code_units
///   trap_if(worst_case_size > MAX_STRING_BYTE_LENGTH)
///   ptr = cx.opts.realloc(0, 0, 2, worst_case_size)
///   trap_if(ptr != align_to(ptr, 2))
///   trap_if(ptr + worst_case_size > len(cx.opts.memory))
///   encoded = src.encode('utf-16-le')
///   cx.opts.memory[ptr : ptr+len(encoded)] = encoded
///   if len(encoded) < worst_case_size:
///     ptr = cx.opts.realloc(ptr, worst_case_size, 2, len(encoded))
///     trap_if(ptr != align_to(ptr, 2))
///     trap_if(ptr + len(encoded) > len(cx.opts.memory))
///   code_units = int(len(encoded) / 2)
///   return (ptr, code_units)
/// ```
pub fn store_utf8_to_utf16<'a, C: Context>(
    _context: &mut C,
    _src: &str,
    _src_code_units: u32,
) -> Result<(u32, u32), ConverterError> {
    todo!(
        "`store_utf8_to_utf16` is not implemented yet because encodings other than UTF-8 are not in scope"
    );
}

/// The next transcoding case handles `latin1+utf16` encoding, where there general goal is to fit the incoming string into Latin-1 if possible based on the code points of the incoming string. The algorithm speculates that all code points *do* fit into Latin-1 and then falls back to a worst-case allocation size when a code point is found outside Latin-1. In this fallback case, the previously-copied Latin-1 bytes are inflated *in place*, inserting a 0 byte after every Latin-1 byte (iterating in reverse to avoid clobbering later bytes):
///
/// ```python
/// def store_string_to_latin1_or_utf16(cx, src, src_code_units):
///   assert(src_code_units <= MAX_STRING_BYTE_LENGTH)
///   ptr = cx.opts.realloc(0, 0, 2, src_code_units)
///   trap_if(ptr != align_to(ptr, 2))
///   trap_if(ptr + src_code_units > len(cx.opts.memory))
///   dst_byte_length = 0
///   for usv in src:
///     if ord(usv) < (1 << 8):
///       cx.opts.memory[ptr + dst_byte_length] = ord(usv)
///       dst_byte_length += 1
///     else:
///       worst_case_size = 2 * src_code_units
///       trap_if(worst_case_size > MAX_STRING_BYTE_LENGTH)
///       ptr = cx.opts.realloc(ptr, src_code_units, 2, worst_case_size)
///       trap_if(ptr != align_to(ptr, 2))
///       trap_if(ptr + worst_case_size > len(cx.opts.memory))
///       for j in range(dst_byte_length-1, -1, -1):
///         cx.opts.memory[ptr + 2*j] = cx.opts.memory[ptr + j]
///         cx.opts.memory[ptr + 2*j + 1] = 0
///       encoded = src.encode('utf-16-le')
///       cx.opts.memory[ptr+2*dst_byte_length : ptr+len(encoded)] = encoded[2*dst_byte_length : ]
///       if worst_case_size > len(encoded):
///         ptr = cx.opts.realloc(ptr, worst_case_size, 2, len(encoded))
///         trap_if(ptr != align_to(ptr, 2))
///         trap_if(ptr + len(encoded) > len(cx.opts.memory))
///       tagged_code_units = int(len(encoded) / 2) | UTF16_TAG
///       return (ptr, tagged_code_units)
///   if dst_byte_length < src_code_units:
///     ptr = cx.opts.realloc(ptr, src_code_units, 2, dst_byte_length)
///     trap_if(ptr != align_to(ptr, 2))
///     trap_if(ptr + dst_byte_length > len(cx.opts.memory))
///   return (ptr, dst_byte_length)
/// ```
pub fn store_string_to_latin1_or_utf16<'a, C: Context>(
    _context: &mut C,
    _src: &str,
    _src_code_units: u32,
) -> Result<(u32, u32), ConverterError> {
    todo!(
        "`store_string_to_latin1_or_utf16` is not implemented yet because encodings other than UTF-8 are not in scope"
    );
}

/// The final transcoding case takes advantage of the extra heuristic information that the incoming UTF-16 bytes were intentionally chosen over Latin-1 by the producer, indicating that they *probably* contain code points outside Latin-1 and thus *probably* require inflation. Based on this information, the transcoding algorithm pessimistically allocates storage for UTF-16, deflating at the end if indeed no non-Latin-1 code points were encountered. This Latin-1 deflation ensures that if a group of components are all using `latin1+utf16` and *one* component over-uses UTF-16, other components can recover the Latin-1 compression. (The Latin-1 check can be inexpensively fused with the UTF-16 validate+copy loop.)
///
/// ```python
/// def store_probably_utf16_to_latin1_or_utf16(cx, src, src_code_units):
///   src_byte_length = 2 * src_code_units
///   trap_if(src_byte_length > MAX_STRING_BYTE_LENGTH)
///   ptr = cx.opts.realloc(0, 0, 2, src_byte_length)
///   trap_if(ptr != align_to(ptr, 2))
///   trap_if(ptr + src_byte_length > len(cx.opts.memory))
///   encoded = src.encode('utf-16-le')
///   cx.opts.memory[ptr : ptr+len(encoded)] = encoded
///   if any(ord(c) >= (1 << 8) for c in src):
///     tagged_code_units = int(len(encoded) / 2) | UTF16_TAG
///     return (ptr, tagged_code_units)
///   latin1_size = int(len(encoded) / 2)
///   for i in range(latin1_size):
///     cx.opts.memory[ptr + i] = cx.opts.memory[ptr + 2*i]
///   ptr = cx.opts.realloc(ptr, src_byte_length, 1, latin1_size)
///   trap_if(ptr + latin1_size > len(cx.opts.memory))
///   return (ptr, latin1_size)
/// ```
pub fn store_probably_utf16_to_latin1_or_utf16<'a, C: Context>(
    _context: &mut C,
    _src: &str,
    _src_code_units: u32,
) -> Result<(u32, u32), ConverterError> {
    todo!(
        "`store_probably_utf16_to_latin1_or_utf16` is not implemented yet because encodings other than UTF-8 are not in scope"
    );
}

/// Error context values are lowered by storing them directly into the per-component-instance `error_contexts` table and passing the `i32` index to wasm.
///
/// ```python
/// def lower_error_context(cx, v):
///   return cx.inst.error_contexts.add(v)
/// ```
pub fn lower_error_context<'a, C: Context>(_context: &mut C, _value: i32) -> i32 {
    todo!(
        "`lower_error_context` is not implemented yet because runtime functionality is not in scope"
    )
}

/// Lists and records are stored by recursively storing their elements and are symmetric to the loading functions. Unlike strings, lists can simply allocate based on the up-front knowledge of length and static element size.
///
/// ```python
/// def store_list(cx, v, ptr, elem_type, maybe_length):
///   if maybe_length is not None:
///     assert(maybe_length == len(v))
///     store_list_into_valid_range(cx, v, ptr, elem_type)
///     return
///   begin, length = store_list_into_range(cx, v, elem_type)
///   store_int(cx, begin, ptr, 4)
///   store_int(cx, length, ptr + 4, 4)
/// ```
///
/// Split into store_list and store_array, depending on whether the length is known or not.
pub fn store_list<'a, C: Context, T: CanonicalAbi>(
    context: &mut C,
    offset: u32,
    value: &[T],
) -> Result<(), ConverterError> {
    let (begin, length) = store_list_into_range(context, value)?;
    store_int!(context, offset, u32, begin);
    store_int!(context, offset + 4, u32, length);
    Ok(())
}

/// See `store_list` for more information.
pub fn store_array<'a, C: Context, T: CanonicalAbi, const LENGTH: u32>(
    context: &mut C,
    offset: u32,
    value: &[T; LENGTH as usize],
) -> Result<(), ConverterError> {
    store_list_into_valid_range(context, offset, value)
}

/// ```python
/// def store_list_into_range(cx, v, elem_type):
///   byte_length = len(v) * elem_size(elem_type)
///   trap_if(byte_length >= (1 << 32))
///   ptr = cx.opts.realloc(0, 0, alignment(elem_type), byte_length)
///   trap_if(ptr != align_to(ptr, alignment(elem_type)))
///   trap_if(ptr + byte_length > len(cx.opts.memory))
///   store_list_into_valid_range(cx, v, ptr, elem_type)
///   return (ptr, len(v))
/// ```
pub fn store_list_into_range<'a, C: Context, T: CanonicalAbi>(
    context: &mut C,
    value: &[T],
) -> Result<(u32, u32), ConverterError> {
    let byte_length = value.len() as u32 * T::SIZE;
    if byte_length >= u32::MAX {
        return Err(ConverterError::ListTooLongToStore { size: byte_length });
    }
    let offset = context.realloc(0, 0, T::ALIGNMENT, byte_length)?;
    if offset == 0 {
        return Err(ConverterError::AllocationFailedForAList);
    }
    if offset != align_to(offset, T::ALIGNMENT) {
        return Err(ConverterError::AllocatedPointerNotAligned);
    }
    store_list_into_valid_range(context, offset, value)?;
    return Ok((offset, value.len() as u32));
}

/// ```python
/// def store_list_into_valid_range(cx, v, ptr, elem_type):
///   for i,e in enumerate(v):
///     store(cx, e, elem_type, ptr + i * elem_size(elem_type))
/// ```
pub fn store_list_into_valid_range<'a, C: Context, T: CanonicalAbi>(
    context: &mut C,
    offset: u32,
    value: &[T],
) -> Result<(), ConverterError> {
    for (i, element) in value.iter().enumerate() {
        store(context, offset + (i as u32 * T::SIZE), element)?;
    }
    Ok(())
}

/// ```python
/// def store_record(cx, v, ptr, fields):
///   for f in fields:
///     ptr = align_to(ptr, alignment(f.t))
///     store(cx, v[f.label], f.t, ptr)
///     ptr += elem_size(f.t)
/// ```
///
/// There also is `store_tuple` for tuples, because we cant treat them like structs in rust.
///
/// Needs to be used in a context where a `ConverterError` can be returned.
#[macro_export]
macro_rules! store_record {
    ($context:expr, $offset:expr, struct $self:path { $($name:ident : $typ:ty,)* }, $value:expr) => {{
        let value = &$value;
        let offset = $offset;
        $(
            let offset = align_to(offset, <$typ>::ALIGNMENT);
            store($context, offset, &value.$name)?;
            #[allow(unused)]
            let offset = offset + <$typ>::SIZE;
        )*
        ()
    }};
}

/// This functions does not have a direct python equivalent, but it does the same as `store_record` but for tuples.
///
/// Needs to be used in a context where a `ConverterError` can be returned.
#[macro_export]
macro_rules! store_tuple {
    ($context:expr, $offset:expr, ( $($typ:ty,)* ), $value:expr) => {{
        let value = &$value;
        let offset = $offset;
        $(
            let offset = align_to(offset, <$typ>::ALIGNMENT);
            store($context, offset, &value.${index()})?;
            #[allow(unused)]
            let offset = offset + <$typ>::SIZE;
        )*
        ()
    }};
}

/// Variant values are represented as Python dictionaries containing exactly one entry whose key is the label of the lifted case and whose value is the (optional) case payload. While this code appears to do an O(n) search of the `variant` type for a matching case label, a normal implementation can statically fuse `store_variant` with its matching `load_variant` to ultimately build a dense array that maps producer's case indices to the consumer's case indices.
///
/// ```python
/// def store_variant(cx, v, ptr, cases):
///   case_index, case_value = match_case(v, cases)
///   disc_size = elem_size(discriminant_type(cases))
///   store_int(cx, case_index, ptr, disc_size)
///   ptr += disc_size
///   ptr = align_to(ptr, max_case_alignment(cases))
///   c = cases[case_index]
///   if c.t is not None:
///     store(cx, case_value, c.t, ptr)
/// ```
#[macro_export]
macro_rules! store_variant {
    ($context:expr, $offset:expr, enum $self:path { $($name:ident $(( $typ:ty ))?,)* }, $value:expr) => {
        {
            #[allow(unused)]
            let value = &$value;
            let offset = $offset;
            const LENGTH: usize = ${count($name)};
            const DISCRIMINATE_SIZE: u32 = crate::discriminant_type::<LENGTH>();

            #[allow(unused_comparisons)]
            match &$value {
                $(<$self>::$name$((value) if size_of::<$typ>() >= 0)? => {
                    match DISCRIMINATE_SIZE {
                        1 => store_int!($context, offset, u8, ${index()} as u8),
                        2 => store_int!($context, offset, u16, ${index()} as u16),
                        4 => store_int!($context, offset, u32, ${index()} as u32),
                        _ => panic!(),
                    }

                    $(
                        // $value
                        #[allow(unused)]
                        let offset = align_to(offset + DISCRIMINATE_SIZE, <$typ>::ALIGNMENT);
                        store($context, offset, value)?;
                    )?
                },)*
                _ => unreachable!(),
            };
        }
    };
}

/// ```python
/// def match_case(v, cases):
///   [label] = v.keys()
///   [index] = [i for i,c in enumerate(cases) if c.label == label]
///   [value] = v.values()
///   return (index, value)
/// ```
///
/// We dont need this function, because Rust does not use dictionaries

/// Flags are converted from a dictionary to a bit-vector by iterating through the case-labels of the variant in the order they were listed in the type definition and OR-ing all the bits together. Flag lifting/lowering can be statically fused into array/integer operations (with a simple byte copy when the case lists are the same) to avoid any string operations in a similar manner to variants.
///
/// ```python
/// def store_flags(cx, v, ptr, labels):
///   i = pack_flags_into_int(v, labels)
///   store_int(cx, i, ptr, elem_size_flags(labels))
/// ```
#[macro_export]
macro_rules! store_flags {
    ($context:expr, $offset:expr, bitflags::Flags<Bits = $ty:ty>, $value:expr) => {{
        let value = pack_flags_into_int!($value, bitflags::Flags<Bits = $ty>);
        store_int!($context, $offset, $ty, value);
    }};
}

/// ```python
/// def pack_flags_into_int(v, labels):
///   i = 0
///   shift = 0
///   for l in labels:
///     i |= (int(bool(v[l])) << shift)
///     shift += 1
///   return i
/// ```
#[macro_export]
macro_rules! pack_flags_into_int {
    ($flags:expr, bitflags::Flags<Bits = $ty:ty>) => {{ $flags.bits() as $ty }};
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WrappedBitflags, tests::SampleContext};

    #[test]
    fn test_store_u8() -> Result<(), ConverterError> {
        let mut context = SampleContext::new(vec![0u8; 4]);
        store_int!(context, 0, u8, 23);
        assert_eq!(context.get_memory(), [23, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_u16() -> Result<(), ConverterError> {
        let mut context = SampleContext::new(vec![0u8; 4]);
        store_int!(context, 0, u16, 257);
        assert_eq!(context.get_memory()[0..4], [1, 1, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_string() -> Result<(), ConverterError> {
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_string(&mut context, 0, "Hello, World!")?;
        assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        assert_eq!(context.get_memory()[4..8], [13, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_list() -> Result<(), ConverterError> {
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_array::<SampleContext, u8, 3>(&mut context, 0, &[1, 2, 3])?;
        assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        assert_eq!(context.get_memory()[4..8], [3, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_struct() -> Result<(), ConverterError> {
        struct SampleStruct {
            a: u8,
            b: u16,
        }
        let val = SampleStruct { a: 3, b: 3 };
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_record!(
            &mut context,
            0,
            struct SampleStruct {
                a: u8,
                b: u16,
            },
            &val
        );
        // assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        // assert_eq!(context.get_memory()[4..8], [3, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_tuple() -> Result<(), ConverterError> {
        let val = (256u16, 65656u32, String::from("Hello, World!"));
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_tuple!(&mut context, 0, (u16, u32, String,), &val);
        // assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        // assert_eq!(context.get_memory()[4..8], [3, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_variant() -> Result<(), ConverterError> {
        #[allow(unused)]
        enum SampleEnum {
            Alpha(u8),
            Beta,
            Gamma(String),
        }
        let val = SampleEnum::Beta;
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_variant!(
            &mut context,
            0,
            enum SampleEnum {
                Alpha(u8),
                Beta,
                Gamma(String),
            },
            &val
        );
        // assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        // assert_eq!(context.get_memory()[4..8], [3, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn test_store_flags() -> Result<(), ConverterError> {
        // use bitflags::bitflags;
        bitflags::bitflags! {
            pub struct Flags: u32 {
                const A = 0b00000001;
                const B = 0b00000010;
                const C = 0b00000100;
            }
        }

        let wrapped: WrappedBitflags<u32, Flags> = WrappedBitflags {
            inner: Flags::A | Flags::C,
        };
        let mut context = SampleContext::new(vec![0u8; 16]);
        store_flags!(&mut context, 0, bitflags::Flags<Bits = u32>, &wrapped.inner);
        // assert_ne!(context.get_memory()[0..4], [0, 0, 0, 0]);
        // assert_eq!(context.get_memory()[4..8], [3, 0, 0, 0]);
        Ok(())
    }
}

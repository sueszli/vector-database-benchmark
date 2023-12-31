/* Copyright © 2017-2023 ABBYY

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#include <common.h>
#pragma hdrstop

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_NEON

#include <CpuMathEngine.h>
#include <CpuExecutionScope.h>
#include <CpuArm.h>
#include <MemoryHandleInternal.h>
#include <MathEngineCommon.h>
#include <MathEngineDnnPoolings.h>

namespace NeoML {

template<class CLoadStore>
static inline void BlobGlobalMaxPoolingWorker( CLoadStore& loadStore, const int32x4_t& iNeon, const float* sourceData,
	int* maxIndexItem, float* resultItem, int maxCount, int resultChannels )
{
	float32x4_t nextVal = loadStore.Load( sourceData );
	int32x4_t nextIndex = iNeon;

	for( int check = 0; check < maxCount; ++check ) {
		float32x4_t curVal = loadStore.Load( resultItem );
		uint32x4_t compare = vcgeq_f32( nextVal, curVal );

		if( !IsMaskZeroNeon( compare ) ) {
			int32x4_t curIndex = loadStore.LoadInt( maxIndexItem );

			float32x4_t curValSet = ConditionNeon( compare, nextVal, curVal );
			loadStore.Store( curValSet, resultItem );
			int32x4_t curIndexSet = ConditionIntNeon( compare, nextIndex, curIndex );
			loadStore.StoreInt( curIndexSet, maxIndexItem );

			if( check < maxCount - 1 ) {
				nextVal = ConditionNeon( compare, curVal, nextVal );
				nextIndex = ConditionIntNeon( compare, curIndex, nextIndex );
			}
		}
		maxIndexItem += resultChannels;
		resultItem += resultChannels;
	}
}

void CCpuMathEngine::BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle& maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices.GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonGlobalMaxPoolingDesc& desc = static_cast<const CCommonGlobalMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int poolSize = source.Height() * source.Width() * source.Depth();
	const int maxCount = result.Height() * result.Width() * result.Depth();

	const float* sourceDataPtr = GetRaw( sourceData );
	CIntHandle maxIndexDataPtr = maxIndices;
	CFloatHandle resultDataPtr = resultData;
	int resultObjectSize = maxCount * result.Channels();

	VectorFill( maxIndexDataPtr, -1, resultObjectSize * result.ObjectCount() );
	VectorFill( resultDataPtr, -FLT_MAX, resultObjectSize * result.ObjectCount() );

	int channels = source.Channels();
	const int channels4 = GetCount4( channels );

	for( int b = 0; b < source.ObjectCount(); ++b ) {
		for( int i = 0; i < poolSize; ++i ) {
			int* maxIndexItem = GetRaw( maxIndexDataPtr );
			float* resultItem = GetRaw( resultDataPtr );

			int32x4_t iNeon = vdupq_n_s32( i );
			for( int c = 0; c < channels4; ++c ) {
				CLoadStoreNeon4 store;
				BlobGlobalMaxPoolingWorker( store, iNeon,
					sourceDataPtr, maxIndexItem, resultItem, maxCount, result.Channels() );

				maxIndexItem += 4;
				resultItem += 4;
				sourceDataPtr += 4;
			}

			if( channels > 0 ) {
				CLoadStoreNeon store( channels );
				BlobGlobalMaxPoolingWorker( store, iNeon, sourceDataPtr, maxIndexItem, resultItem, maxCount, result.Channels() );
				sourceDataPtr += channels;
			}
		}
		maxIndexDataPtr += resultObjectSize;
		resultDataPtr += resultObjectSize;
	}
}

//---------------------------------------------------------------------------------------------------
// 3d max pooling
static inline void blob3dMaxPoolingProcessFirstItem( const float* sourceObject, int sourceIndex,
	int channels4, int channels, float* resultData, int* indexData )
{
	const float* sourceData = sourceObject + sourceIndex;

	int32x4_t sourceIndexNeon = vdupq_n_s32( sourceIndex );
	for( int c = 0; c < channels4; ++c ) {
		StoreNeon4( LoadNeon4( sourceData ), resultData );
		StoreIntNeon4( sourceIndexNeon, indexData );

		sourceData += 4;
		resultData += 4;
		indexData += 4;
	}

	if( channels > 0 ) {
		StoreNeon( LoadNeon( sourceData, channels ), resultData, channels );
		StoreIntNeon( sourceIndexNeon, indexData, channels );
	}
}

static inline void blob3dMaxPoolingProcessFirstItem( const float* sourceObject, int sourceIndex,
	int channels4, int channels, float* resultData )
{
	const float* sourceData = sourceObject + sourceIndex;

	for( int c = 0; c < channels4; ++c ) {
		StoreNeon4( LoadNeon4( sourceData ), resultData );
		sourceData += 4;
		resultData += 4;
	}

	if( channels > 0 ) {
		StoreNeon( LoadNeon( sourceData, channels ), resultData, channels );
	}
}

static inline void blob3dMaxPoolingProcessItem( const float* sourceObject, int sourceIndex,
	int channels4, int channels, float* resultData, int* indexData )
{
	const float* sourceData = sourceObject + sourceIndex;

	int32x4_t sourceIndexNeon = vdupq_n_s32( sourceIndex );
	for( int c = 0; c < channels4; ++c ) {
		float32x4_t res = LoadNeon4( resultData );
		float32x4_t src = LoadNeon4( sourceData );
		int32x4_t index = LoadIntNeon4( indexData );

		uint32x4_t cmp = vcltq_f32( res, src );

		res = ConditionNeon( cmp, src, res );
		StoreNeon4( res, resultData );

		index = ConditionNeon( cmp, sourceIndexNeon, index );
		StoreIntNeon4( index, indexData );

		sourceData += 4;
		resultData += 4;
		indexData += 4;
	}

	for( int c = 0; c < channels; ++c ) {
		if( *resultData < *sourceData ) {
			*resultData = *sourceData;
			*indexData = sourceIndex;
		}
		++sourceData;
		++resultData;
		++indexData;
	}
}

static inline void blob3dMaxPoolingProcessItem( const float* sourceObject, int sourceIndex,
	int channels4, int channels, float* resultData )
{
	const float* sourceData = sourceObject + sourceIndex;

	for( int c = 0; c < channels4; ++c ) {
		float32x4_t res = LoadNeon4( resultData );
		float32x4_t src = LoadNeon4( sourceData );
		StoreNeon4( vmaxq_f32( res, src ), resultData );
		sourceData += 4;
		resultData += 4;
	}

	if( channels > 0 ) {
		float32x4_t res = LoadNeon( resultData, channels );
		float32x4_t src = LoadNeon( sourceData, channels );
		StoreNeon( vmaxq_f32( res, src ), resultData, channels );
	}
}

void CCpuMathEngine::Blob3dMaxPooling( const C3dMaxPoolingDesc& poolingDesc, const CConstFloatHandle& sourceData,
	const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommon3dMaxPoolingDesc& desc = static_cast<const CCommon3dMaxPoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const float* sourceObject = GetRaw( sourceData );
	float* resultJStart = GetRaw( resultData );
	int* indexJStart = ( maxIndices == 0 ) ? 0 : GetRaw( *maxIndices );

	const int sourceDepthSize = source.Depth() * source.Channels();
	const int sourceRowSize = source.Width() * sourceDepthSize;
	const int sourceObjectSize = source.Height() * sourceRowSize;

	const int resultDepthSize = result.Depth() * result.Channels();
	const int resultRowSize = result.Width() * resultDepthSize;

	int channels = result.Channels();
	const int channels4 = GetCount4( channels );

	for( int b = 0; b < result.ObjectCount(); ++b ) {
		// Go through all cube blocks and iterate through each block
		// So we will get a forward pass over the source and filterHeight * filterWidth passes over the result
		for( int j = 0; j < result.Height(); ++j ) {
			int sourceJIndex = j * desc.StrideHeight * sourceRowSize;
			for( int filterJ = 0; filterJ < desc.FilterHeight; ++filterJ ) {
				float* resultIStart = resultJStart;
				int* indexIStart = indexJStart;

				for( int i = 0; i < result.Width(); ++i ) {
					int sourceIIndex = sourceJIndex + i * desc.StrideWidth * sourceDepthSize;
					for( int filterI = 0; filterI < desc.FilterWidth; ++filterI ) {
						float* resultDataPtr = resultIStart;
						int* indexData = indexIStart;

						for( int k = 0; k < result.Depth(); ++k ) {
							int sourceIndex = sourceIIndex + k * desc.StrideDepth * source.Channels();
							for( int filterK = 0; filterK < desc.FilterDepth; ++filterK ) {
								if( ( filterJ == 0 ) && ( filterI == 0 ) && ( filterK == 0 ) ) {
									if( indexData == 0 ) {
										blob3dMaxPoolingProcessFirstItem( sourceObject, sourceIndex,
											channels4, channels, resultDataPtr );
									} else {
										blob3dMaxPoolingProcessFirstItem( sourceObject, sourceIndex,
											channels4, channels, resultDataPtr, indexData );
									}
								} else {
									if( indexData == 0 ) {
										blob3dMaxPoolingProcessItem( sourceObject, sourceIndex,
											channels4, channels, resultDataPtr );
									} else {
										blob3dMaxPoolingProcessItem( sourceObject, sourceIndex,
											channels4, channels, resultDataPtr, indexData );
									}
								}
								sourceIndex += source.Channels();
							}
							resultDataPtr += result.Channels();
							if( indexData != 0 ) {
								indexData += result.Channels();
							}
						}
						sourceIIndex += sourceDepthSize;
					}
					resultIStart += resultDepthSize;
					if( indexIStart != 0 ) {
						indexIStart += resultDepthSize;
					}
				}
				sourceJIndex += sourceRowSize;
			}
			resultJStart += resultRowSize;
			if( indexJStart != 0 ) {
				indexJStart += resultRowSize;
			}
		}
		sourceObject += sourceObjectSize;
	}
}

//---------------------------------------------------------------------------------------------------

void CCpuMathEngine::BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& poolingDesc,
	const CConstFloatHandle& sourceData, const CIntHandle* maxIndices, const CFloatHandle& resultData )
{
	ASSERT_EXPR( sourceData.GetMathEngine() == this );
	ASSERT_EXPR( maxIndices == 0 || maxIndices->GetMathEngine() == this );
	ASSERT_EXPR( resultData.GetMathEngine() == this );
	CCpuExecutionScope scope;

	const CCommonMaxOverTimePoolingDesc& desc = static_cast<const CCommonMaxOverTimePoolingDesc&>( poolingDesc );
	const CBlobDesc& source = desc.Source;
	const CBlobDesc& result = desc.Result;

	const int seqElemTotalSize = source.ObjectSize() * source.BatchWidth();
	int seqElemSize = seqElemTotalSize;
	const int seqElemSize4 = GetCount4( seqElemSize );

	CConstFloatHandle sourceStart = sourceData;
	CFloatHandle resultStart = resultData;

	if( maxIndices != 0 ) {
		CIntHandle indexStart = *maxIndices;

		int indexValueStart = 0;
		for( int l = 0; l < result.BatchLength(); ++l ) {
			const float* sourceDataPtr = GetRaw( sourceStart );

			// Set the initial values ("zero value")
			CCpuMathEngine::VectorCopy( resultStart, sourceStart, seqElemTotalSize );
			CCpuMathEngine::VectorFill( indexStart, indexValueStart, seqElemTotalSize );

			sourceDataPtr += seqElemTotalSize;

			for( int n = 1; n < desc.FilterLen; ++n ) {
				// Restart the result data
				float* resultDataPtr = GetRaw( resultStart );
				int* indexData = GetRaw( indexStart );
				const int indexValue = indexValueStart + n;
				int32x4_t indexValueNeon = vdupq_n_s32( indexValue );

				for( int i = 0; i < seqElemSize4; ++i ) {
					float32x4_t src = LoadNeon4( sourceDataPtr );
					float32x4_t res = LoadNeon4( resultDataPtr );
					int32x4_t cmp = vcgtq_f32( src, res );

					res = vmaxq_f32( res, src );
					StoreNeon4( res, resultDataPtr );

					int32x4_t ind = LoadIntNeon4( indexData );
					ind = ConditionIntNeon( cmp, indexValueNeon, ind );
					StoreIntNeon4( ind, indexData );

					sourceDataPtr += 4;
					resultDataPtr += 4;
					indexData += 4;
				}

				for( int i = 0; i < seqElemSize; ++i ) {
					if( *sourceDataPtr > *resultDataPtr ) {
						*resultDataPtr = *sourceDataPtr;
						*indexData = indexValue;
					}
					++sourceDataPtr;
					++resultDataPtr;
					++indexData;
				}
			}

			sourceStart += desc.StrideLen * seqElemTotalSize;
			resultStart += seqElemTotalSize;
			indexStart += seqElemTotalSize;

			indexValueStart += desc.StrideLen;
		}

	} else {
		for( int l = 0; l < result.BatchLength(); ++l ) {
			const float* sourceDataPtr = GetRaw( sourceStart );

			// Set the initial values ("zero value")
			CCpuMathEngine::VectorCopy( resultStart, sourceStart, seqElemTotalSize );

			sourceDataPtr += seqElemTotalSize;

			for( int n = 1; n < desc.FilterLen; ++n ) {
				// Restart the result data
				float* resultDataPtr = GetRaw( resultStart );

				for( int i = 0; i < seqElemSize4; ++i ) {
					float32x4_t src = LoadNeon4( sourceDataPtr );
					float32x4_t res = LoadNeon4( resultDataPtr );

					res = vmaxq_f32( res, src );
					StoreNeon4( res, resultDataPtr );

					sourceDataPtr += 4;
					resultDataPtr += 4;
				}

				if( seqElemSize > 0 ) {
					float32x4_t src = LoadNeon( sourceDataPtr, seqElemSize );
					float32x4_t res = LoadNeon( resultDataPtr, seqElemSize );

					res = vmaxq_f32( res, src );
					StoreNeon( res, resultDataPtr, seqElemSize );

					sourceDataPtr += seqElemSize;
				}
			}

			sourceStart += desc.StrideLen * seqElemTotalSize;
			resultStart += seqElemTotalSize;
		}
	}
}

} // namespace NeoML

#endif // NEOML_USE_NEON

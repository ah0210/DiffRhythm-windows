import sys
import os
sys.path.append(os.getcwd())

from infer.lightweight_infer import LightweightMusicGenerator
from infer.ai_api_infer import AIApiManager, AIApiMusicGenerator

def test_lightweight_model():
    """测试轻量级模型生成"""
    print("=== 测试轻量级模型生成 ===")
    lightweight_gen = LightweightMusicGenerator(device='cpu')
    
    try:
        # 加载轻量级模型
        lightweight_gen.load_lightweight_model()
        print("✅ 轻量级模型加载成功")
        
        # 测试1: 使用文本提示生成
        print("\n🔤 测试使用文本提示生成:")
        output_path = lightweight_gen.generate_with_lightweight_model(
            text_prompt="Electronic Dance Music",
            duration=3,
            steps=5
        )
        print(f"✅ 文本提示生成成功，输出文件: {output_path}")
        
        # 测试2: 检查生成的文件是否为有效的WAV文件
        print("\n🔍 测试生成的WAV文件有效性:")
        import os
        import wave
        if os.path.exists(output_path):
            try:
                with wave.open(output_path, 'rb') as wf:
                    print(f"✅ WAV文件有效")
                    print(f"   • 声道数: {wf.getnchannels()}")
                    print(f"   • 采样宽度: {wf.getsampwidth()} bytes")
                    print(f"   • 采样率: {wf.getframerate()} Hz")
                    print(f"   • 帧数: {wf.getnframes()}")
                    print(f"   • 时长: {wf.getnframes() / wf.getframerate():.2f} 秒")
            except Exception as e:
                print(f"❌ WAV文件无效: {e}")
                return False
        else:
            print(f"❌ WAV文件不存在: {output_path}")
            return False
        
        # 测试3: 使用音频提示生成（模拟）
        print("\n🎧 测试使用音频提示生成:")
        # 创建一个临时的空WAV文件作为模拟音频提示
        temp_wav_path = "temp_test.wav"
        with wave.open(temp_wav_path, 'wb') as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b'')
        
        output_path2 = lightweight_gen.generate_with_lightweight_model(
            wav_path=temp_wav_path,
            duration=3,
            steps=5
        )
        print(f"✅ 音频提示生成成功，输出文件: {output_path2}")
        
        # 清理临时文件
        os.remove(temp_wav_path)
        
        return True
    except Exception as e:
        print(f"❌ 轻量级模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_api():
    """测试AI API生成"""
    print("\n=== 测试AI API生成 ===")
    ai_gen = AIApiMusicGenerator(device='cpu')
    
    try:
        # 使用免费API生成音乐
        output_path = ai_gen.generate(
            text_prompt="Pop Music",
            duration=10,
            api_provider="free"
        )
        print(f"✅ AI API生成成功，输出文件: {output_path}")
        return True
    except Exception as e:
        print(f"❌ AI API测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_manager():
    """测试API管理器"""
    print("\n=== 测试API管理器 ===")
    api_manager = AIApiManager()
    
    try:
        # 获取可用的API提供商
        providers = api_manager.get_available_providers()
        print(f"✅ 可用的API提供商: {providers}")
        
        # 获取API提供商信息
        for provider in providers:
            info = api_manager.get_provider_info(provider)
            print(f"   • {provider}: {info['name']} - {info['description']}")
        
        return True
    except Exception as e:
        print(f"❌ API管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 轻量级模型和AI API测试 ===")
    
    # 运行测试
    lightweight_result = test_lightweight_model()
    api_result = test_ai_api()
    api_manager_result = test_api_manager()
    
    print("\n=== 测试结果 ===")
    print(f"轻量级模型测试: {'通过' if lightweight_result else '失败'}")
    print(f"AI API测试: {'通过' if api_result else '失败'}")
    print(f"API管理器测试: {'通过' if api_manager_result else '失败'}")
    
    if lightweight_result and api_result and api_manager_result:
        print("✅ 所有测试通过！")
        sys.exit(0)
    else:
        print("❌ 部分测试失败！")
        sys.exit(1)
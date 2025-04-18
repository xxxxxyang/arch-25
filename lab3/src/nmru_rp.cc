/**
 * Copyright (c) 2018-2020 Inria
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer;
 * redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution;
 * neither the name of the copyright holders nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

 #include "mem/cache/replacement_policies/nmru_rp.hh"

 #include <cassert>
 #include <memory>
 
 #include "params/NMRURP.hh"
 #include "sim/cur_tick.hh"
 #include "base/random.hh"
 
 namespace gem5
 {
 
 GEM5_DEPRECATED_NAMESPACE(ReplacementPolicy, replacement_policy);
 namespace replacement_policy
 {
 
 NMRU::NMRU(const Params &p)
   : Base(p)
 {
 }
 
 void
 NMRU::invalidate(const std::shared_ptr<ReplacementData>& replacement_data)
 {
     // Reset last touch timestamp
     std::static_pointer_cast<NMRUReplData>(
         replacement_data)->lastTouchTick = Tick(0);
 }
 
 void
 NMRU::touch(const std::shared_ptr<ReplacementData>& replacement_data) const
 {
     // Update last touch timestamp
     std::static_pointer_cast<NMRUReplData>(
         replacement_data)->lastTouchTick = curTick();
 }
 
 void
 NMRU::reset(const std::shared_ptr<ReplacementData>& replacement_data) const
 {
     // Set last touch timestamp
     std::static_pointer_cast<NMRUReplData>(
         replacement_data)->lastTouchTick = curTick();
 }
 
 ReplaceableEntry*
 NMRU::getVictim(const ReplacementCandidates& candidates) const
 {
     // There must be at least one replacement candidate
     assert(candidates.size() > 0);
 
     // Visit all candidates to find victim
     ReplaceableEntry* mru = candidates[0];

     // find the most recently used entry
     for (const auto& candidate : candidates) {
         if (std::static_pointer_cast<NMRUReplData>(
                 candidate->replacementData)->lastTouchTick >
             std::static_pointer_cast<NMRUReplData>(
                 mru->replacementData)->lastTouchTick) {
             mru = candidate;
         }
     }
      // randomly get a victim from the candidates except the most recently used
      std::vector<ReplaceableEntry*> nmru_candidates;
      for (const auto& candidate : candidates) {
          if (candidate != mru) {
              nmru_candidates.push_back(candidate);
          }
      }
      if (nmru_candidates.empty()) {
          return mru;
      }
      ReplaceableEntry* victim = nmru_candidates[random_mt.random<unsigned>(0,
                                      nmru_candidates.size() - 1)];
 
     return victim;
 }
 
 std::shared_ptr<ReplacementData>
 NMRU::instantiateEntry()
 {
     return std::shared_ptr<ReplacementData>(new NMRUReplData());
 }
 
 } // namespace replacement_policy
 } // namespace gem5
 